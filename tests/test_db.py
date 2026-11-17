"""Unit tests for database configuration and session management (db.py)."""

import os
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import db as db_module

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5453/model_registry",
)


class TestGetEngine:
    """Tests for get_engine()."""

    def test_get_engine_creates_and_caches(self) -> None:
        """get_engine creates a real engine on first call and returns the same instance on second call."""
        db_module._engine = None
        db_module._async_session_factory = None
        try:
            engine = db_module.get_engine()
            # Verify real engine was created with expected URL from test env settings
            assert engine.url.drivername == "postgresql+psycopg"

            # Second call returns the exact same engine object (caching)
            engine2 = db_module.get_engine()
            assert engine2 is engine
        finally:
            db_module._engine = None
            db_module._async_session_factory = None


class TestGetSessionFactory:
    """Tests for get_session_factory()."""

    def test_get_session_factory_creates_and_caches(self) -> None:
        """get_session_factory creates a factory on first call and caches it."""
        db_module._engine = None
        db_module._async_session_factory = None
        try:
            factory = db_module.get_session_factory()
            # Verify the factory is a real async_sessionmaker bound to a real engine
            assert factory.class_ is AsyncSession

            # Second call returns the exact same factory object (caching)
            factory2 = db_module.get_session_factory()
            assert factory2 is factory
        finally:
            db_module._engine = None
            db_module._async_session_factory = None


class TestGetSession:
    """Tests for get_session() async generator."""

    async def test_get_session_yields_functional_session(self) -> None:
        """get_session yields a functional AsyncSession that can execute queries."""
        db_module._engine = None
        db_module._async_session_factory = None
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

            test_engine = create_async_engine(TEST_DATABASE_URL)
            db_module._engine = test_engine
            db_module._async_session_factory = async_sessionmaker(
                test_engine, expire_on_commit=False, class_=AsyncSession
            )

            gen = db_module.get_session()
            session = await gen.__anext__()
            # Verify the session is a real AsyncSession (not a mock)
            assert isinstance(session, AsyncSession)
            # Verify the session can actually execute a query against the test DB
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            # A second query reuses the same session/connection successfully,
            # proving the yielded session is fully functional.
            result2 = await session.execute(text("SELECT 2"))
            assert result2.scalar() == 2
            # The generator must complete after yielding exactly one session
            # (this drives the `async with factory() as sess:` cleanup path).
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()
        finally:
            if db_module._engine is not None and hasattr(db_module._engine, "dispose"):
                await db_module._engine.dispose()
            db_module._engine = None
            db_module._async_session_factory = None


class TestInitDb:
    """Tests for init_db()."""

    async def test_init_db_creates_tables(self) -> None:
        """init_db creates all tables from ORM metadata."""
        db_module._engine = None
        db_module._async_session_factory = None
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            test_engine = create_async_engine(TEST_DATABASE_URL)
            db_module._engine = test_engine

            await db_module.init_db()

            # Verify tables were created
            async with test_engine.connect() as conn:

                def check_tables(sync_conn):
                    from sqlalchemy import inspect

                    inspector = inspect(sync_conn)
                    return inspector.get_table_names()

                tables = await conn.run_sync(check_tables)
                assert "model_versions" in tables
                assert "routing_rules" in tables
        finally:
            if db_module._engine is not None:
                await db_module._engine.dispose()
            db_module._engine = None
            db_module._async_session_factory = None


class TestAdditiveMigrations:
    """Tests for ADDITIVE_MIGRATIONS / apply_additive_migrations.

    Simulates a legacy DB where some recent ORM additions were never
    persisted (drop the columns / tables after create_all, then re-run
    init_db) and asserts every gap is closed. This pins the contract
    that protects long-running production DBs from silent ORM/DB drift.
    """

    async def test_additive_migrations_restore_dropped_columns_and_tables(self, engine) -> None:
        """init_db's additive pass restores deploy_* columns and the alias/tag/audit tables."""
        from sqlalchemy import inspect, text

        # Simulate a legacy DB: drop everything the additive pass owns.
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE model_versions DROP COLUMN deployed_at"))
            await conn.execute(text("ALTER TABLE model_versions DROP COLUMN deploy_status"))
            await conn.execute(text("ALTER TABLE model_versions DROP COLUMN deploy_error"))
            await conn.execute(text("ALTER TABLE model_versions DROP COLUMN deploy_started_at"))
            await conn.execute(text("ALTER TABLE model_versions DROP COLUMN deploy_completed_at"))
            await conn.execute(text("DROP TABLE model_aliases"))
            await conn.execute(text("DROP TABLE model_tags"))
            await conn.execute(text("DROP TABLE audit_events"))

        # Sanity: the legacy state is what we expect.
        async with engine.connect() as conn:

            def _snapshot(sync_conn):
                ins = inspect(sync_conn)
                cols = {c["name"] for c in ins.get_columns("model_versions")}
                tables = set(ins.get_table_names())
                return cols, tables

            cols_before, tables_before = await conn.run_sync(_snapshot)
        assert "deployed_at" not in cols_before
        assert "deploy_status" not in cols_before
        assert "model_aliases" not in tables_before
        assert "model_tags" not in tables_before
        assert "audit_events" not in tables_before

        # Run the additive migrations directly against the legacy engine.
        await db_module.apply_additive_migrations(engine)

        async with engine.connect() as conn:
            cols_after, tables_after = await conn.run_sync(_snapshot)

        # Columns restored.
        for name in (
            "deployed_at",
            "deploy_status",
            "deploy_error",
            "deploy_started_at",
            "deploy_completed_at",
        ):
            assert name in cols_after, f"column {name} missing after additive migration"
        # Tables restored.
        for name in ("model_aliases", "model_tags", "audit_events"):
            assert name in tables_after, f"table {name} missing after additive migration"

        # Indexes that the ORM declares — verify the named ones the additive
        # pass owns. SQLAlchemy auto-generated index names use ``ix_<table>_<col>``.
        async with engine.connect() as conn:

            def _indexes(sync_conn, table: str) -> set[str]:
                ins = inspect(sync_conn)
                return {ix["name"] for ix in ins.get_indexes(table)}

            audit_idx = await conn.run_sync(_indexes, "audit_events")
            alias_idx = await conn.run_sync(_indexes, "model_aliases")
            tag_idx = await conn.run_sync(_indexes, "model_tags")
        assert "ix_audit_entity_type_id" in audit_idx
        assert "ix_audit_type_timestamp" in audit_idx
        assert "ix_model_aliases_model_id" in alias_idx
        assert "ix_model_tags_key_value" in tag_idx

    async def test_additive_migrations_idempotent(self, engine) -> None:
        """Running additive migrations twice in a row succeeds (no-op second pass)."""
        # First pass — create_all in the fixture already covers the schema,
        # so this is a true idempotency test against an already-current DB.
        await db_module.apply_additive_migrations(engine)
        # Second pass must not raise.
        await db_module.apply_additive_migrations(engine)

    async def test_additive_migrations_writes_smoke(self, engine) -> None:
        """After the additive pass, INSERTs into the restored shape succeed.

        Catches type/constraint mismatches between the ADDITIVE DDL and the
        ORM (e.g. wrong NOT NULL, wrong default) that a column-name check
        alone would miss.
        """
        from sqlalchemy import text

        # Drop and re-create via additive pass to mirror the legacy path.
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE model_aliases"))
            await conn.execute(text("DROP TABLE model_tags"))
            await conn.execute(text("DROP TABLE audit_events"))
        await db_module.apply_additive_migrations(engine)

        async with engine.begin() as conn:
            # audit_events: actor has DEFAULT 'system', detail JSONB nullable.
            await conn.execute(
                text(
                    "INSERT INTO audit_events "
                    "(event_type, entity_type, entity_id) "
                    "VALUES ('test.event', 'model_version', 'mv-1')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO model_aliases "
                    "(schema_id, alias_name, model_id) "
                    "VALUES ('s1', 'champion', "
                    "'00000000-0000-0000-0000-000000000001'::uuid)"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO model_tags "
                    "(model_id, key, value) "
                    "VALUES ('00000000-0000-0000-0000-000000000001'::uuid, 'env', 'prod')"
                )
            )

        async with engine.connect() as conn:
            audit_actor = (
                await conn.execute(text("SELECT actor FROM audit_events LIMIT 1"))
            ).scalar()
            alias_count = (await conn.execute(text("SELECT count(*) FROM model_aliases"))).scalar()
            tag_count = (await conn.execute(text("SELECT count(*) FROM model_tags"))).scalar()
        assert audit_actor == "system"  # default applied
        assert alias_count == 1
        assert tag_count == 1


class TestCloseDb:
    """Tests for close_db()."""

    async def test_close_db_disposes_engine(self) -> None:
        """close_db disposes the engine and resets the singletons."""
        from sqlalchemy.ext.asyncio import create_async_engine

        test_engine = create_async_engine(TEST_DATABASE_URL)
        db_module._engine = test_engine
        db_module._async_session_factory = MagicMock()

        await db_module.close_db()

        assert db_module._engine is None
        assert db_module._async_session_factory is None

    async def test_close_db_when_no_engine(self) -> None:
        """close_db does nothing when there is no engine."""
        db_module._engine = None
        db_module._async_session_factory = None
        await db_module.close_db()
        assert db_module._engine is None


class TestStringListJSON:
    """Tests for the StringListJSON custom SQLAlchemy type decorator."""

    def _make_type(self):
        from models.db import StringListJSON

        return StringListJSON()

    def _mock_dialect(self):
        return MagicMock()

    def test_process_bind_param_none(self) -> None:
        """process_bind_param returns None when value is None."""
        t = self._make_type()
        result = t.process_bind_param(None, self._mock_dialect())
        assert result is None

    def test_process_bind_param_list(self) -> None:
        """process_bind_param returns the list unchanged."""
        t = self._make_type()
        result = t.process_bind_param(["a", "b"], self._mock_dialect())
        assert result == ["a", "b"]

    def test_process_result_value_none(self) -> None:
        """process_result_value returns None when value is None."""
        t = self._make_type()
        result = t.process_result_value(None, self._mock_dialect())
        assert result is None

    def test_process_result_value_string(self) -> None:
        """process_result_value parses a JSON string into a list."""
        t = self._make_type()
        result = t.process_result_value('["a","b"]', self._mock_dialect())
        assert result == ["a", "b"]

    def test_process_result_value_list(self) -> None:
        """process_result_value returns a list when given a list."""
        t = self._make_type()
        result = t.process_result_value(["a", "b"], self._mock_dialect())
        assert result == ["a", "b"]


class TestORMRepr:
    """Tests for __repr__ methods on all ORM models."""

    def test_model_version_repr(self) -> None:
        """ModelVersionORM.__repr__ contains id, schema_id, version, and status."""
        from uuid import UUID

        from models.db import ModelVersionORM

        obj = ModelVersionORM(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            schema_id="my_schema",
            version="1.0",
            status="active",
        )
        r = repr(obj)
        assert "12345678-1234-5678-1234-567812345678" in r
        assert "my_schema" in r
        assert "1.0" in r
        assert "active" in r

    def test_routing_rule_repr(self) -> None:
        """RoutingRuleORM.__repr__ contains schema_id and active_model_id."""
        from uuid import UUID

        from models.db import RoutingRuleORM

        obj = RoutingRuleORM(
            schema_id="route_schema",
            active_model_id=UUID("abcdefab-abcd-abcd-abcd-abcdefabcdef"),
        )
        r = repr(obj)
        assert "route_schema" in r
        assert "abcdefab-abcd-abcd-abcd-abcdefabcdef" in r

    def test_audit_event_repr(self) -> None:
        """AuditEventORM.__repr__ contains id, event_type, entity_type, and entity_id."""
        from models.db import AuditEventORM

        obj = AuditEventORM(
            id=42,
            event_type="model.promoted",
            entity_type="model_version",
            entity_id="ent-123",
        )
        r = repr(obj)
        assert "42" in r
        assert "model.promoted" in r
        assert "model_version" in r
        assert "ent-123" in r

    def test_model_alias_repr(self) -> None:
        """ModelAliasORM.__repr__ contains schema_id, alias_name, and model_id."""
        from uuid import UUID

        from models.db import ModelAliasORM

        obj = ModelAliasORM(
            schema_id="alias_schema",
            alias_name="champion",
            model_id=UUID("11111111-2222-3333-4444-555555555555"),
        )
        r = repr(obj)
        assert "alias_schema" in r
        assert "champion" in r
        assert "11111111-2222-3333-4444-555555555555" in r

    def test_model_tag_repr(self) -> None:
        """ModelTagORM.__repr__ contains model_id, key, and value."""
        from uuid import UUID

        from models.db import ModelTagORM

        obj = ModelTagORM(
            model_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            key="env",
            value="production",
        )
        r = repr(obj)
        assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" in r
        assert "env" in r
        assert "production" in r
