"""Tests for POST /api/flywheel/run endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app


@asynccontextmanager
async def _noop_lifespan(app: Any) -> AsyncIterator[None]:
    yield


@pytest.fixture
def flywheel_client():
    """Yield a lightweight async client (no DB needed for flywheel tests)."""

    async def _make() -> AsyncClient:
        app = create_app(custom_lifespan=_noop_lifespan)
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _make


class TestFlywheelRunEndpoint:
    """Tests for POST /api/flywheel/run."""

    async def test_accepts_valid_request(self, flywheel_client) -> None:
        """Valid request returns 202 Accepted with correct body."""
        async with await flywheel_client() as client:
            with patch(
                "routers.flywheel.run_flywheel_subprocess",
                new_callable=AsyncMock,
                return_value={"status": "completed", "detail": "ok"},
            ):
                resp = await client.post(
                    "/api/flywheel/run",
                    json={
                        "schema_id": "ks-entity-spotter",
                        "batch_id": "batch-abc-123",
                        "minio_path": "batches/ks-entity-spotter/batch-abc-123",
                    },
                )

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["schema_id"] == "ks-entity-spotter"
        assert "batch-abc-123" in body["message"]

    async def test_accepts_custom_min_samples(self, flywheel_client) -> None:
        """Custom min_samples value is accepted."""
        async with await flywheel_client() as client:
            with patch(
                "routers.flywheel.run_flywheel_subprocess",
                new_callable=AsyncMock,
                return_value={"status": "completed", "detail": "ok"},
            ):
                resp = await client.post(
                    "/api/flywheel/run",
                    json={
                        "schema_id": "ks-signal-extractor",
                        "batch_id": "batch-xyz",
                        "minio_path": "batches/ks-signal-extractor/batch-xyz",
                        "min_samples": 50,
                    },
                )

        assert resp.status_code == 202
        assert resp.json()["schema_id"] == "ks-signal-extractor"

    async def test_rejects_missing_schema_id(self, flywheel_client) -> None:
        """Missing schema_id returns 422."""
        async with await flywheel_client() as client:
            resp = await client.post(
                "/api/flywheel/run",
                json={
                    "batch_id": "batch-abc",
                    "minio_path": "batches/test/batch-abc",
                },
            )
        assert resp.status_code == 422

    async def test_rejects_missing_batch_id(self, flywheel_client) -> None:
        """Missing batch_id returns 422."""
        async with await flywheel_client() as client:
            resp = await client.post(
                "/api/flywheel/run",
                json={
                    "schema_id": "ks-entity-spotter",
                    "minio_path": "batches/test/batch-abc",
                },
            )
        assert resp.status_code == 422

    async def test_rejects_missing_minio_path(self, flywheel_client) -> None:
        """Missing minio_path returns 422."""
        async with await flywheel_client() as client:
            resp = await client.post(
                "/api/flywheel/run",
                json={
                    "schema_id": "ks-entity-spotter",
                    "batch_id": "batch-abc",
                },
            )
        assert resp.status_code == 422

    async def test_rejects_invalid_min_samples(self, flywheel_client) -> None:
        """min_samples < 1 returns 422."""
        async with await flywheel_client() as client:
            resp = await client.post(
                "/api/flywheel/run",
                json={
                    "schema_id": "ks-entity-spotter",
                    "batch_id": "batch-abc",
                    "minio_path": "batches/test/batch-abc",
                    "min_samples": 0,
                },
            )
        assert resp.status_code == 422

    async def test_rejects_empty_body(self, flywheel_client) -> None:
        """Empty JSON body returns 422."""
        async with await flywheel_client() as client:
            resp = await client.post("/api/flywheel/run", json={})
        assert resp.status_code == 422

    async def test_background_task_called_with_correct_args(self, flywheel_client) -> None:
        """Verify the subprocess is invoked with the right parameters."""
        mock_subprocess = AsyncMock(return_value={"status": "completed", "detail": "ok"})

        async with await flywheel_client() as client:
            with patch(
                "routers.flywheel.run_flywheel_subprocess",
                mock_subprocess,
            ):
                resp = await client.post(
                    "/api/flywheel/run",
                    json={
                        "schema_id": "ks-test-schema",
                        "batch_id": "batch-999",
                        "minio_path": "batches/ks-test-schema/batch-999",
                        "min_samples": 30,
                    },
                )

        assert resp.status_code == 202
        mock_subprocess.assert_called_once_with(
            schema_id="ks-test-schema",
            batch_id="batch-999",
            minio_path="batches/ks-test-schema/batch-999",
            min_samples=30,
        )


class TestFlywheelServiceSubprocess:
    """Tests for the flywheel_service subprocess wrapper."""

    async def test_subprocess_success(self) -> None:
        """Successful subprocess returns completed status."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"all good\n", None))

        with patch(
            "services.flywheel_service.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            from services.flywheel_service import run_flywheel_subprocess

            result = await run_flywheel_subprocess(
                schema_id="test-schema",
                batch_id="batch-1",
                minio_path="batches/test-schema/batch-1",
            )

        assert result["status"] == "completed"

    async def test_subprocess_failure(self) -> None:
        """Non-zero exit code returns failed status with tail output."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"line1\nline2\nERROR: training failed\n", None)
        )

        with patch(
            "services.flywheel_service.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            from services.flywheel_service import run_flywheel_subprocess

            result = await run_flywheel_subprocess(
                schema_id="test-schema",
                batch_id="batch-1",
                minio_path="batches/test-schema/batch-1",
            )

        assert result["status"] == "failed"
        assert "Exit code 1" in result["detail"]
        assert "training failed" in result["detail"]

    async def test_subprocess_exception(self) -> None:
        """Exception during subprocess launch returns failed status."""
        with patch(
            "services.flywheel_service.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("python not found"),
        ):
            from services.flywheel_service import run_flywheel_subprocess

            result = await run_flywheel_subprocess(
                schema_id="test-schema",
                batch_id="batch-1",
                minio_path="batches/test-schema/batch-1",
            )

        assert result["status"] == "failed"
        assert "python not found" in result["detail"]

    async def test_subprocess_timeout_kills_child(self) -> None:
        """A subprocess that runs past the timeout is killed and reported as failed.

        The test patches the module-level timeout to a tiny value and replaces
        ``proc.communicate`` with a coroutine that sleeps forever so
        ``asyncio.wait_for`` fires a ``TimeoutError``. We then assert:

        1. ``proc.kill()`` was called exactly once (no leaked child).
        2. ``proc.wait()`` was awaited to reap the killed child.
        3. The returned dict reports a failed status whose detail mentions
           the configured timeout value.
        """

        async def never_returns() -> tuple[bytes, bytes]:
            # Sleep well past any plausible wait_for timeout. asyncio will
            # cancel this task when the wait_for timeout elapses.
            await asyncio.sleep(60)
            return (b"", b"")  # pragma: no cover - cancelled before returning

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.communicate = never_returns
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with (
            patch(
                "services.flywheel_service.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
            patch(
                "services.flywheel_service._FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS",
                0.05,
            ),
        ):
            from services.flywheel_service import run_flywheel_subprocess

            result = await run_flywheel_subprocess(
                schema_id="test-schema",
                batch_id="batch-timeout",
                minio_path="batches/test-schema/batch-timeout",
            )

        assert result["status"] == "failed"
        assert "timeout" in result["detail"].lower()
        assert "killed" in result["detail"].lower()
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    async def test_subprocess_real_sleep_timeout(self) -> None:
        """End-to-end timeout test against a real ``sleep 10`` child process.

        Unlike ``test_subprocess_timeout_kills_child`` (which mocks the proc
        object), this test actually spawns ``/bin/sleep 10`` via the real
        ``asyncio.create_subprocess_exec`` and relies on the wait_for /
        kill + wait path to reap it. That covers the real OS-level
        interaction that the mocked test cannot.
        """
        import shutil

        # Point the flywheel script at an executable that sleeps so we don't
        # execute anything heavy; we only care that the timeout fires.
        sleep_bin = shutil.which("sleep")
        if sleep_bin is None:  # pragma: no cover - all CI runners have sleep
            pytest.skip("sleep binary not available")

        # Capture the real unpatched create_subprocess_exec so we can bypass
        # the patch below and spawn a real child process without recursion.
        real_create_subprocess_exec = asyncio.create_subprocess_exec

        async def fake_create(*args, **kwargs):
            return await real_create_subprocess_exec(
                sleep_bin,
                "10",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        with (
            patch(
                "services.flywheel_service.asyncio.create_subprocess_exec",
                side_effect=fake_create,
            ),
            patch(
                "services.flywheel_service._FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS",
                0.2,
            ),
        ):
            from services.flywheel_service import run_flywheel_subprocess

            result = await run_flywheel_subprocess(
                schema_id="test-schema",
                batch_id="batch-real-timeout",
                minio_path="batches/test-schema/batch-real-timeout",
            )

        assert result["status"] == "failed"
        assert "timeout" in result["detail"].lower()
