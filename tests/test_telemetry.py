"""Unit tests for telemetry.py (OTEL setup with graceful degradation)."""

import socket
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

import telemetry
from telemetry import _test_connection, setup_telemetry


@pytest.fixture(autouse=True)
def _reset_initialized():
    """Save and restore the telemetry._initialized global between tests."""
    original = telemetry._initialized
    telemetry._initialized = False
    yield
    telemetry._initialized = original


@pytest.fixture()
def _enable_otel(monkeypatch: pytest.MonkeyPatch):
    """Ensure OTEL_SDK_DISABLED is unset so setup_telemetry proceeds."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)


@pytest.fixture()
def mock_settings(monkeypatch: pytest.MonkeyPatch):
    """Provide a mock get_settings that returns a fake otlp_traces_endpoint."""
    mock_s = MagicMock()
    mock_s.otlp_traces_endpoint = "http://grafana-agent:4318/v1/traces"
    mock_get = MagicMock(return_value=mock_s)
    return mock_get


# ---------------------------------------------------------------------------
# _test_connection
# ---------------------------------------------------------------------------
class TestTestConnection:
    """Tests for the _test_connection helper."""

    def test_connection_success(self) -> None:
        """Returns True when TCP connection succeeds."""
        mock_sock = MagicMock()
        with patch.object(socket, "create_connection", return_value=mock_sock) as mock_create:
            result = _test_connection("localhost", 4318)

        assert result is True
        mock_create.assert_called_once_with(("localhost", 4318), timeout=2.0)
        mock_sock.close.assert_called_once()

    def test_connection_timeout(self) -> None:
        """Returns False when connection times out."""
        with patch.object(socket, "create_connection", side_effect=TimeoutError):
            result = _test_connection("localhost", 4318, timeout=1.0)

        assert result is False

    def test_connection_refused(self) -> None:
        """Returns False when connection is refused."""
        with patch.object(socket, "create_connection", side_effect=ConnectionRefusedError):
            result = _test_connection("localhost", 4318)

        assert result is False

    def test_connection_os_error(self) -> None:
        """Returns False on generic OSError (e.g. DNS failure)."""
        with patch.object(socket, "create_connection", side_effect=OSError("Network unreachable")):
            result = _test_connection("unreachable-host", 4318)

        assert result is False


# ---------------------------------------------------------------------------
# setup_telemetry
# ---------------------------------------------------------------------------
class TestSetupTelemetry:
    """Tests for the setup_telemetry function."""

    def test_already_initialized(self) -> None:
        """Returns True immediately when telemetry is already initialized."""
        telemetry._initialized = True
        app = FastAPI()

        result = setup_telemetry(app)

        assert result is True

    def test_otel_sdk_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when OTEL_SDK_DISABLED=true."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        app = FastAPI()

        result = setup_telemetry(app)

        assert result is False
        assert telemetry._initialized is False

    def test_otel_packages_import_fails(self, _enable_otel: None) -> None:
        """Returns False when OTEL packages cannot be imported."""
        app = FastAPI()

        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Force the import inside setup_telemetry to fail
            with patch("builtins.__import__", side_effect=_make_import_blocker("opentelemetry")):
                result = setup_telemetry(app)

        assert result is False
        assert telemetry._initialized is False

    def test_full_success(self, _enable_otel: None, mock_settings: MagicMock) -> None:
        """Full success path: connection succeeds, all instrumentation works."""
        app = FastAPI()

        mock_resource = MagicMock()
        mock_provider = MagicMock()
        mock_exporter = MagicMock()
        mock_processor = MagicMock()
        mock_fastapi_instrumentor = MagicMock()
        mock_httpx_instrumentor = MagicMock()

        with (
            patch("telemetry._test_connection", return_value=True),
            patch("telemetry.os.getenv", return_value="false"),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter": MagicMock(),
                    "opentelemetry.exporter.otlp": MagicMock(),
                    "opentelemetry.exporter.otlp.proto": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock(return_value=mock_exporter)
                    ),
                    "opentelemetry.instrumentation": MagicMock(),
                    "opentelemetry.instrumentation.fastapi": MagicMock(
                        FastAPIInstrumentor=mock_fastapi_instrumentor
                    ),
                    "opentelemetry.instrumentation.httpx": MagicMock(
                        HTTPXClientInstrumentor=MagicMock(return_value=mock_httpx_instrumentor)
                    ),
                    "opentelemetry.sdk": MagicMock(),
                    "opentelemetry.sdk.resources": MagicMock(
                        SERVICE_NAME="service.name",
                        Resource=MagicMock(create=MagicMock(return_value=mock_resource)),
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider)
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(
                        BatchSpanProcessor=MagicMock(return_value=mock_processor)
                    ),
                    "settings": MagicMock(get_settings=mock_settings),
                },
            ),
        ):
            result = setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True
        mock_provider.add_span_processor.assert_called_once_with(mock_processor)
        mock_fastapi_instrumentor.instrument_app.assert_called_once_with(app)

    def test_endpoint_unreachable(self, _enable_otel: None, mock_settings: MagicMock) -> None:
        """When endpoint is unreachable, still returns True but skips exporter."""
        app = FastAPI()

        mock_resource = MagicMock()
        mock_provider = MagicMock()

        with (
            patch("telemetry._test_connection", return_value=False),
            patch("telemetry.os.getenv", return_value="false"),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter": MagicMock(),
                    "opentelemetry.exporter.otlp": MagicMock(),
                    "opentelemetry.exporter.otlp.proto": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock()
                    ),
                    "opentelemetry.instrumentation": MagicMock(),
                    "opentelemetry.instrumentation.fastapi": MagicMock(
                        FastAPIInstrumentor=MagicMock()
                    ),
                    "opentelemetry.instrumentation.httpx": MagicMock(
                        HTTPXClientInstrumentor=MagicMock(return_value=MagicMock())
                    ),
                    "opentelemetry.sdk": MagicMock(),
                    "opentelemetry.sdk.resources": MagicMock(
                        SERVICE_NAME="service.name",
                        Resource=MagicMock(create=MagicMock(return_value=mock_resource)),
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider)
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
                    "settings": MagicMock(get_settings=mock_settings),
                },
            ),
        ):
            result = setup_telemetry(app)

        assert result is True
        assert telemetry._initialized is True
        # Exporter NOT added because endpoint was unreachable
        mock_provider.add_span_processor.assert_not_called()

    def test_exporter_setup_exception(self, _enable_otel: None, mock_settings: MagicMock) -> None:
        """When exporter setup raises, logs warning and continues."""
        app = FastAPI()

        mock_resource = MagicMock()
        mock_provider = MagicMock()

        # Make urlparse raise inside the try block
        mock_settings_obj = mock_settings()
        mock_settings_obj.otlp_traces_endpoint = "http://grafana-agent:4318/v1/traces"
        bad_mock_settings = MagicMock(return_value=mock_settings_obj)

        with (
            patch("telemetry._test_connection", side_effect=OSError("parse boom")),
            patch("telemetry.os.getenv", return_value="false"),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter": MagicMock(),
                    "opentelemetry.exporter.otlp": MagicMock(),
                    "opentelemetry.exporter.otlp.proto": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock()
                    ),
                    "opentelemetry.instrumentation": MagicMock(),
                    "opentelemetry.instrumentation.fastapi": MagicMock(
                        FastAPIInstrumentor=MagicMock()
                    ),
                    "opentelemetry.instrumentation.httpx": MagicMock(
                        HTTPXClientInstrumentor=MagicMock(return_value=MagicMock())
                    ),
                    "opentelemetry.sdk": MagicMock(),
                    "opentelemetry.sdk.resources": MagicMock(
                        SERVICE_NAME="service.name",
                        Resource=MagicMock(create=MagicMock(return_value=mock_resource)),
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider)
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
                    "settings": MagicMock(get_settings=bad_mock_settings),
                },
            ),
        ):
            result = setup_telemetry(app)

        # Still returns True -- exporter failure is non-fatal
        assert result is True
        assert telemetry._initialized is True

    def test_fastapi_instrumentation_fails(
        self, _enable_otel: None, mock_settings: MagicMock
    ) -> None:
        """When FastAPI instrumentation raises, logs warning and continues."""
        app = FastAPI()

        mock_resource = MagicMock()
        mock_provider = MagicMock()
        mock_fastapi_instrumentor = MagicMock()
        mock_fastapi_instrumentor.instrument_app.side_effect = RuntimeError("FastAPI boom")

        with (
            patch("telemetry._test_connection", return_value=False),
            patch("telemetry.os.getenv", return_value="false"),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter": MagicMock(),
                    "opentelemetry.exporter.otlp": MagicMock(),
                    "opentelemetry.exporter.otlp.proto": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock()
                    ),
                    "opentelemetry.instrumentation": MagicMock(),
                    "opentelemetry.instrumentation.fastapi": MagicMock(
                        FastAPIInstrumentor=mock_fastapi_instrumentor
                    ),
                    "opentelemetry.instrumentation.httpx": MagicMock(
                        HTTPXClientInstrumentor=MagicMock(return_value=MagicMock())
                    ),
                    "opentelemetry.sdk": MagicMock(),
                    "opentelemetry.sdk.resources": MagicMock(
                        SERVICE_NAME="service.name",
                        Resource=MagicMock(create=MagicMock(return_value=mock_resource)),
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider)
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
                    "settings": MagicMock(get_settings=mock_settings),
                },
            ),
        ):
            result = setup_telemetry(app)

        # Still returns True -- FastAPI instrumentation failure is non-fatal
        assert result is True
        assert telemetry._initialized is True

    def test_httpx_instrumentation_fails(
        self, _enable_otel: None, mock_settings: MagicMock
    ) -> None:
        """When HTTPX instrumentation raises, logs warning and continues."""
        app = FastAPI()

        mock_resource = MagicMock()
        mock_provider = MagicMock()
        mock_httpx_instrumentor = MagicMock()
        mock_httpx_instrumentor.return_value.instrument.side_effect = RuntimeError("HTTPX boom")

        with (
            patch("telemetry._test_connection", return_value=False),
            patch("telemetry.os.getenv", return_value="false"),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter": MagicMock(),
                    "opentelemetry.exporter.otlp": MagicMock(),
                    "opentelemetry.exporter.otlp.proto": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock()
                    ),
                    "opentelemetry.instrumentation": MagicMock(),
                    "opentelemetry.instrumentation.fastapi": MagicMock(
                        FastAPIInstrumentor=MagicMock()
                    ),
                    "opentelemetry.instrumentation.httpx": MagicMock(
                        HTTPXClientInstrumentor=mock_httpx_instrumentor
                    ),
                    "opentelemetry.sdk": MagicMock(),
                    "opentelemetry.sdk.resources": MagicMock(
                        SERVICE_NAME="service.name",
                        Resource=MagicMock(create=MagicMock(return_value=mock_resource)),
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider)
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
                    "settings": MagicMock(get_settings=mock_settings),
                },
            ),
        ):
            result = setup_telemetry(app)

        # Still returns True -- HTTPX instrumentation failure is non-fatal
        assert result is True
        assert telemetry._initialized is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_import_blocker(blocked_prefix: str):
    """Return an __import__ side_effect that blocks imports starting with the given prefix.

    All other imports delegate to the real built-in __import__.
    """
    import builtins

    real_import = builtins.__import__

    def _blocker(name, *args, **kwargs):
        if name.startswith(blocked_prefix):
            raise ImportError(f"Mocked import failure for {name}")
        return real_import(name, *args, **kwargs)

    return _blocker
