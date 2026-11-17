"""Tests for flywheel adapter/model version pruning."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from minio.error import S3Error

# Add scripts directory to path so we can import flywheel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import flywheel  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minio_object(name: str, last_modified: datetime | None = None) -> SimpleNamespace:
    """Create a fake MinIO object with the given name and timestamp."""
    return SimpleNamespace(
        object_name=name,
        last_modified=last_modified or datetime.now(UTC),
        size=1024,
        etag="abc123",
    )


def _make_version_objects(
    schema_id: str, version_id: str, ts: datetime, file_count: int = 3
) -> list[SimpleNamespace]:
    """Create fake MinIO objects for an adapter version directory."""
    files = ["adapter_config.json", "adapter_model.safetensors", "README.md"]
    return [_make_minio_object(f"{schema_id}/{version_id}/{f}", ts) for f in files[:file_count]]


# ---------------------------------------------------------------------------
# _discover_versions tests
# ---------------------------------------------------------------------------


class TestDiscoverVersions:
    """Tests for _discover_versions."""

    def test_empty_bucket(self):
        mock_client = MagicMock()
        mock_client.list_objects.return_value = iter([])

        result = flywheel._discover_versions(mock_client, "adapters", "ks-entity-spotter/")
        assert result == []

    def test_single_version(self):
        now = datetime.now(UTC)
        mock_client = MagicMock()
        objects = _make_version_objects("ks-entity-spotter", "v1", now)
        mock_client.list_objects.return_value = iter(objects)

        result = flywheel._discover_versions(mock_client, "adapters", "ks-entity-spotter/")
        assert len(result) == 1
        assert result[0][0] == "v1"
        assert result[0][1] == now

    def test_multiple_versions_sorted_newest_first(self):
        now = datetime.now(UTC)
        old = now - timedelta(days=7)
        older = now - timedelta(days=14)

        mock_client = MagicMock()
        objects = (
            _make_version_objects("schema1", "v3", now)
            + _make_version_objects("schema1", "v1", older)
            + _make_version_objects("schema1", "v2", old)
        )
        mock_client.list_objects.return_value = iter(objects)

        result = flywheel._discover_versions(mock_client, "adapters", "schema1/")
        assert len(result) == 3
        assert [v[0] for v in result] == ["v3", "v2", "v1"]

    def test_uses_newest_object_timestamp_per_version(self):
        """If a version has files with different timestamps, use the newest."""
        now = datetime.now(UTC)
        old = now - timedelta(hours=1)

        mock_client = MagicMock()
        objects = [
            _make_minio_object("s/v1/file1.json", old),
            _make_minio_object("s/v1/file2.bin", now),
        ]
        mock_client.list_objects.return_value = iter(objects)

        result = flywheel._discover_versions(mock_client, "adapters", "s/")
        assert len(result) == 1
        assert result[0][1] == now


# ---------------------------------------------------------------------------
# _delete_version tests
# ---------------------------------------------------------------------------


class TestDeleteVersion:
    """Tests for _delete_version."""

    def test_deletes_all_objects_under_prefix(self):
        mock_client = MagicMock()
        objects = [
            _make_minio_object("schema/v1/file1"),
            _make_minio_object("schema/v1/file2"),
            _make_minio_object("schema/v1/file3"),
        ]
        mock_client.list_objects.return_value = iter(objects)

        count = flywheel._delete_version(mock_client, "adapters", "schema/v1/")
        assert count == 3
        assert mock_client.remove_object.call_count == 3

    def test_returns_zero_for_empty_prefix(self):
        mock_client = MagicMock()
        mock_client.list_objects.return_value = iter([])

        count = flywheel._delete_version(mock_client, "adapters", "nonexistent/")
        assert count == 0
        mock_client.remove_object.assert_not_called()


# ---------------------------------------------------------------------------
# _get_minio_client tests
# ---------------------------------------------------------------------------


class TestGetMinioClient:
    """Tests for _get_minio_client."""

    def test_raises_without_credentials(self):
        with (
            patch.object(flywheel, "MINIO_ACCESS_KEY", ""),
            patch.object(flywheel, "MINIO_SECRET_KEY", ""),
        ):
            with pytest.raises(RuntimeError, match="MinIO credentials not configured"):
                flywheel._get_minio_client()

    def test_creates_client_with_credentials(self):
        with (
            patch.object(flywheel, "MINIO_ACCESS_KEY", "testkey"),
            patch.object(flywheel, "MINIO_SECRET_KEY", "testsecret"),
            patch.object(flywheel, "MINIO_ENDPOINT", "localhost:9000"),
            patch.object(flywheel, "MINIO_SECURE", False),
        ):
            client = flywheel._get_minio_client()
            assert isinstance(client, flywheel.Minio)


# ---------------------------------------------------------------------------
# _get_active_adapter_paths tests
# ---------------------------------------------------------------------------


class TestGetActiveAdapterPaths:
    """Tests for _get_active_adapter_paths."""

    @pytest.mark.asyncio
    async def test_returns_active_paths(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"schema_id": "s1", "adapter_path": "s1/v5/"},
            {"schema_id": "s2", "adapter_path": "s2/v3/"},
            {"schema_id": "s3", "adapter_path": None},
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = MagicMock(return_value=mock_response)

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        result = await flywheel._get_active_adapter_paths(mock_client)
        assert result == {"s1/v5/", "s2/v3/"}

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = mock_get

        result = await flywheel._get_active_adapter_paths(mock_client)
        assert result == set()


# ---------------------------------------------------------------------------
# prune_adapter_versions tests
# ---------------------------------------------------------------------------


class TestPruneAdapterVersions:
    """Tests for prune_adapter_versions."""

    @pytest.mark.asyncio
    async def test_prunes_old_versions_keeps_latest_two(self):
        now = datetime.now(UTC)
        v1_ts = now - timedelta(days=14)
        v2_ts = now - timedelta(days=7)
        v3_ts = now

        mock_minio = MagicMock()

        # list_objects called multiple times:
        # 1. Top-level schema listing (recursive=False)
        # 2. Per-schema version discovery (recursive=True)
        # 3. Per-version delete listing (recursive=True)
        top_level_objects = [_make_minio_object("ks-entity-spotter/")]

        v1_objects = _make_version_objects("ks-entity-spotter", "v1", v1_ts)
        v2_objects = _make_version_objects("ks-entity-spotter", "v2", v2_ts)
        v3_objects = _make_version_objects("ks-entity-spotter", "v3", v3_ts)
        all_version_objects = v1_objects + v2_objects + v3_objects

        delete_v1_objects = _make_version_objects("ks-entity-spotter", "v1", v1_ts)

        call_count = {"n": 0}

        def mock_list_objects(bucket, prefix="", recursive=False):
            call_count["n"] += 1
            if not recursive:
                return iter(top_level_objects)
            if prefix == "ks-entity-spotter/v1/":
                return iter(delete_v1_objects)
            return iter(all_version_objects)

        mock_minio.list_objects = mock_list_objects

        # Mock routing response (v3 is active)
        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = [
                {"schema_id": "ks-entity-spotter", "adapter_path": "ks-entity-spotter/v3/"}
            ]
            resp.raise_for_status = MagicMock()
            return resp

        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.get = mock_get

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            results = await flywheel.prune_adapter_versions(mock_http, dry_run=False, keep=2)

        assert "ks-entity-spotter" in results
        assert results["ks-entity-spotter"] == 1  # v1 deleted
        # Verify v1 objects were deleted
        assert mock_minio.remove_object.call_count == 3  # 3 files in v1

    @pytest.mark.asyncio
    async def test_dry_run_does_not_delete(self):
        now = datetime.now(UTC)
        v1_ts = now - timedelta(days=14)
        v2_ts = now - timedelta(days=7)
        v3_ts = now

        mock_minio = MagicMock()

        top_level_objects = [_make_minio_object("schema1/")]
        all_version_objects = (
            _make_version_objects("schema1", "v1", v1_ts)
            + _make_version_objects("schema1", "v2", v2_ts)
            + _make_version_objects("schema1", "v3", v3_ts)
        )

        def mock_list_objects(bucket, prefix="", recursive=False):
            if not recursive:
                return iter(top_level_objects)
            return iter(all_version_objects)

        mock_minio.list_objects = mock_list_objects

        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            return resp

        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.get = mock_get

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            results = await flywheel.prune_adapter_versions(mock_http, dry_run=True, keep=2)

        assert results["schema1"] == 1  # Would delete v1
        mock_minio.remove_object.assert_not_called()  # But didn't actually delete

    @pytest.mark.asyncio
    async def test_protects_active_deployed_version(self):
        """Even if an active version is old, it must not be deleted."""
        now = datetime.now(UTC)
        v1_ts = now - timedelta(days=30)
        v2_ts = now - timedelta(days=20)
        v3_ts = now - timedelta(days=10)
        v4_ts = now

        mock_minio = MagicMock()

        top_level_objects = [_make_minio_object("schema1/")]
        all_version_objects = (
            _make_version_objects("schema1", "v1", v1_ts)
            + _make_version_objects("schema1", "v2", v2_ts)
            + _make_version_objects("schema1", "v3", v3_ts)
            + _make_version_objects("schema1", "v4", v4_ts)
        )

        def mock_list_objects(bucket, prefix="", recursive=False):
            if not recursive:
                return iter(top_level_objects)
            if "v1/" in prefix:
                return iter(_make_version_objects("schema1", "v1", v1_ts))
            return iter(all_version_objects)

        mock_minio.list_objects = mock_list_objects

        # v2 is the deployed/active version (even though it's old)
        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = [{"schema_id": "schema1", "adapter_path": "schema1/v2/"}]
            resp.raise_for_status = MagicMock()
            return resp

        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.get = mock_get

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            results = await flywheel.prune_adapter_versions(mock_http, dry_run=False, keep=2)

        # v4 and v3 are kept (latest 2), v2 is protected (active), only v1 is deleted
        assert results.get("schema1", 0) == 1

    @pytest.mark.asyncio
    async def test_nothing_to_prune(self):
        mock_minio = MagicMock()

        now = datetime.now(UTC)
        top_level_objects = [_make_minio_object("schema1/")]
        all_version_objects = _make_version_objects(
            "schema1", "v1", now - timedelta(days=1)
        ) + _make_version_objects("schema1", "v2", now)

        def mock_list_objects(bucket, prefix="", recursive=False):
            if not recursive:
                return iter(top_level_objects)
            return iter(all_version_objects)

        mock_minio.list_objects = mock_list_objects

        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            return resp

        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.get = mock_get

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            results = await flywheel.prune_adapter_versions(mock_http, dry_run=False, keep=2)

        assert results == {}
        mock_minio.remove_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_adapters_bucket(self):
        mock_minio = MagicMock()
        mock_minio.list_objects.return_value = iter([])

        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            return resp

        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.get = mock_get

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            results = await flywheel.prune_adapter_versions(mock_http)

        assert results == {}


# ---------------------------------------------------------------------------
# prune_old_models tests
# ---------------------------------------------------------------------------


class TestPruneOldModels:
    """Tests for prune_old_models."""

    @pytest.mark.asyncio
    async def test_keeps_base_model_deletes_others(self):
        mock_minio = MagicMock()

        top_level = [
            _make_minio_object("Qwen3-4B-AWQ-270m-it/"),
            _make_minio_object("old-model-v1/"),
            _make_minio_object("old-model-v2/"),
        ]

        old_model_objects = [
            _make_minio_object("old-model-v1/model.bin"),
            _make_minio_object("old-model-v1/config.json"),
        ]

        def mock_list_objects(bucket, prefix="", recursive=False):
            if not recursive and not prefix:
                return iter(top_level)
            if prefix == "old-model-v1/":
                return iter(old_model_objects)
            if prefix == "old-model-v2/":
                return iter([_make_minio_object("old-model-v2/model.bin")])
            return iter([])

        mock_minio.list_objects = mock_list_objects

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            deleted = await flywheel.prune_old_models(dry_run=False)

        assert deleted == 2  # old-model-v1 and old-model-v2

    @pytest.mark.asyncio
    async def test_dry_run_does_not_delete_models(self):
        mock_minio = MagicMock()

        top_level = [
            _make_minio_object("Qwen3-4B-AWQ-270m-it/"),
            _make_minio_object("garbage/"),
        ]

        def mock_list_objects(bucket, prefix="", recursive=False):
            return iter(top_level)

        mock_minio.list_objects = mock_list_objects

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            deleted = await flywheel.prune_old_models(dry_run=True)

        assert deleted == 1  # Would delete garbage/
        mock_minio.remove_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_models_bucket(self):
        mock_minio = MagicMock()
        mock_minio.list_objects.side_effect = S3Error(
            MagicMock(), "NoSuchBucket", "NoSuchBucket", "", "", ""
        )

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            deleted = await flywheel.prune_old_models(dry_run=False)

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_keeps_every_protected_model_dir(self):
        """Regression: the 2026-04-17 incident where prune_old_models deleted
        gemma-4-e2b-it-text every 4am UTC because only Qwen3-4B-AWQ-270m-it
        was protected. Now all names in PROTECTED_MODEL_DIRS must survive.
        """
        mock_minio = MagicMock()

        top_level = [_make_minio_object(f"{name}/") for name in flywheel.PROTECTED_MODEL_DIRS] + [
            _make_minio_object("old-finetune/")
        ]

        def mock_list_objects(bucket, prefix="", recursive=False):
            if not recursive and not prefix:
                return iter(top_level)
            if prefix == "old-finetune/":
                return iter([_make_minio_object("old-finetune/model.bin")])
            return iter([])

        mock_minio.list_objects = mock_list_objects

        with patch.object(flywheel, "_get_minio_client", return_value=mock_minio):
            deleted = await flywheel.prune_old_models(dry_run=False)

        assert deleted == 1  # only old-finetune
        # None of the protected dirs were ever passed to remove_object's prefix
        removed_objects = [c.args[1] for c in mock_minio.remove_object.call_args_list]
        for protected in flywheel.PROTECTED_MODEL_DIRS:
            assert not any(obj.startswith(f"{protected}/") for obj in removed_objects), (
                f"{protected} should be protected but was deleted"
            )


# ---------------------------------------------------------------------------
# run_prune integration test
# ---------------------------------------------------------------------------


class TestRunPrune:
    """Tests for the run_prune orchestrator."""

    @pytest.mark.asyncio
    async def test_runs_both_adapter_and_model_pruning(self):
        with (
            patch.object(flywheel, "prune_adapter_versions", return_value={}) as mock_adapters,
            patch.object(flywheel, "prune_old_models", return_value=0) as mock_models,
        ):
            mock_http = MagicMock(spec=httpx.AsyncClient)
            result = await flywheel.run_prune(mock_http, dry_run=True, keep=3)

        assert result is True
        mock_adapters.assert_called_once_with(mock_http, dry_run=True, keep=3)
        mock_models.assert_called_once_with(dry_run=True)

    @pytest.mark.asyncio
    async def test_returns_false_on_adapter_failure(self):
        with (
            patch.object(flywheel, "prune_adapter_versions", side_effect=RuntimeError("boom")),
            patch.object(flywheel, "prune_old_models", return_value=0),
        ):
            mock_http = MagicMock(spec=httpx.AsyncClient)
            result = await flywheel.run_prune(mock_http)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_model_failure(self):
        with (
            patch.object(flywheel, "prune_adapter_versions", return_value={}),
            patch.object(flywheel, "prune_old_models", side_effect=RuntimeError("boom")),
        ):
            mock_http = MagicMock(spec=httpx.AsyncClient)
            result = await flywheel.run_prune(mock_http)

        assert result is False
