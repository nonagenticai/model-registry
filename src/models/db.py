"""SQLAlchemy ORM models for the model registry.

These models mirror the Pydantic schemas in models/domain.py and provide
persistence to PostgreSQL via psycopg.

Column types are chosen for cross-database compatibility (PostgreSQL in
production, SQLite in tests):
- UUID is stored as Uuid (SQLAlchemy 2.0 native, auto-adapts per dialect)
- ARRAY is stored as JSON (works on both PG and SQLite)
- JSONB is stored as JSON (auto-adapts to JSONB on PG)
"""

import json as _json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

from db import Base
from settings import DEFAULT_BASE_MODEL, DEFAULT_FALLBACK_CHAIN, DEFAULT_QUANTIZATION


class StringListJSON(TypeDecorator):
    """Store a list[str] as JSON text.

    On PostgreSQL this serialises to native JSONB; on SQLite it stores as TEXT.
    The value is always returned as a Python list[str].
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: list[str] | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        return value

    def process_result_value(self, value: Any, dialect: Dialect) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return _json.loads(value)
        return list(value)


class ModelVersionORM(Base):
    """ORM model for a registered model version."""

    __tablename__ = "model_versions"

    id: Mapped[Any] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid7)
    schema_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="training", index=True)
    gguf_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    gguf_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adapter_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    model_format: Mapped[str] = mapped_column(String(16), nullable=False, default="gguf")
    base_model: Mapped[str] = mapped_column(String(256), nullable=False, default=DEFAULT_BASE_MODEL)
    quantization: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DEFAULT_QUANTIZATION
    )
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deploy_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    deploy_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    deploy_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deploy_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_model_versions_schema_status", "schema_id", "status"),
        Index("ix_model_versions_schema_version", "schema_id", "version", unique=True),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion {self.id} schema={self.schema_id} v={self.version} [{self.status}]>"


class RoutingRuleORM(Base):
    """ORM model for a routing rule mapping schema_id to its active model."""

    __tablename__ = "routing_rules"

    schema_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Content-addressable fingerprint of the response_format JSON schema.
    # NULL for rules created before auto-registration landed (legacy rules
    # that were hand-PUT with a human-readable schema_id). For auto-
    # registered rules the fingerprint is unique — first caller to submit
    # a given schema wins the uuid7 and everyone else reuses it.
    schema_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True, unique=True, index=True
    )
    active_model_id: Mapped[Any | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    fallback_chain: Mapped[list[str] | None] = mapped_column(
        StringListJSON(),
        nullable=False,
        default=lambda: list(DEFAULT_FALLBACK_CHAIN),
    )
    endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    json_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    gbnf_grammar: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    adapter_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    adapter_loaded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=sa.text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return f"<RoutingRule schema={self.schema_id} model={self.active_model_id}>"


class AuditEventORM(Base):
    """ORM model for audit trail events."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(256), nullable=False, default="system")
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_audit_entity_type_id", "entity_type", "entity_id"),
        Index("ix_audit_type_timestamp", "event_type", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AuditEvent {self.id} {self.event_type} {self.entity_type}:{self.entity_id}>"


class ModelAliasORM(Base):
    """ORM model for named model aliases (e.g. champion, rollback)."""

    __tablename__ = "model_aliases"

    schema_id: Mapped[str] = mapped_column(String(256), nullable=False, primary_key=True)
    alias_name: Mapped[str] = mapped_column(String(64), nullable=False, primary_key=True)
    model_id: Mapped[Any] = mapped_column(Uuid(as_uuid=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return f"<ModelAlias {self.schema_id}:{self.alias_name} -> {self.model_id}>"


class ModelTagORM(Base):
    """ORM model for key-value tags on model versions."""

    __tablename__ = "model_tags"

    model_id: Mapped[Any] = mapped_column(Uuid(as_uuid=True), nullable=False, primary_key=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, primary_key=True)
    value: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (Index("ix_model_tags_key_value", "key", "value"),)

    def __repr__(self) -> str:
        return f"<ModelTag {self.model_id} {self.key}={self.value}>"


class GlobalAdapterORM(Base):
    """ORM model for the system-wide default LoRA adapter.

    Single-row table enforced by ``CHECK (id = 1)``. Holds the adapter the
    inference-router falls through to when no enabled per-schema rule
    matches. Phase 1 substrate; populated by model-trainer in Phase 2.
    """

    __tablename__ = "global_adapter"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        nullable=False,
        default=1,
        server_default=sa.text("1"),
    )
    model_version_id: Mapped[Any] = mapped_column(Uuid(as_uuid=True), nullable=False)
    minio_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    adapter_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )

    __table_args__ = (CheckConstraint("id = 1", name="ck_global_adapter_singleton"),)

    def __repr__(self) -> str:
        return f"<GlobalAdapter mv={self.model_version_id} hash={self.adapter_hash}>"
