"""Unit tests for logging configuration (logging_config.py)."""

from unittest.mock import MagicMock, patch

import logging_config as log_module


class TestAddTraceContext:
    """Tests for add_trace_context processor."""

    def test_without_otel(self) -> None:
        """Without OTEL, event_dict is returned unchanged."""
        event_dict = {"event": "test"}
        # When OTEL_AVAILABLE is False, no trace_id is added
        with patch.object(log_module, "OTEL_AVAILABLE", False):
            result = log_module.add_trace_context(None, "info", event_dict)
        assert "trace_id" not in result
        assert "span_id" not in result

    def test_with_otel_no_recording_span(self) -> None:
        """With OTEL available but no recording span, no trace context is added."""
        event_dict = {"event": "test"}
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(log_module, "OTEL_AVAILABLE", True),
            patch.object(log_module, "trace", mock_trace),
        ):
            result = log_module.add_trace_context(None, "info", event_dict)

        assert "trace_id" not in result

    def test_with_otel_recording_span(self) -> None:
        """With an active recording span, trace_id and span_id are injected."""
        event_dict = {"event": "test"}
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 0x0123456789ABCDEF0123456789ABCDEF
        mock_ctx.span_id = 0x0123456789ABCDEF
        mock_ctx.is_valid = True

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(log_module, "OTEL_AVAILABLE", True),
            patch.object(log_module, "trace", mock_trace),
        ):
            result = log_module.add_trace_context(None, "info", event_dict)

        assert "trace_id" in result
        assert "span_id" in result

    def test_with_otel_invalid_context(self) -> None:
        """With OTEL but invalid span context, no trace info is added."""
        event_dict = {"event": "test"}
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(log_module, "OTEL_AVAILABLE", True),
            patch.object(log_module, "trace", mock_trace),
        ):
            result = log_module.add_trace_context(None, "info", event_dict)

        assert "trace_id" not in result


class TestAddServiceContext:
    """Tests for add_service_context processor."""

    def test_adds_service_and_environment(self) -> None:
        """add_service_context adds service and environment fields."""
        event_dict = {"event": "test"}
        result = log_module.add_service_context(None, "info", event_dict)
        assert result["service"] == "model-registry"
        assert "environment" in result


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_configure_logging_json_format(self) -> None:
        """configure_logging with json format configures structlog."""
        log_module._configured = False
        try:
            mock_settings = MagicMock()
            mock_settings.log_format = "json"
            mock_settings.log_level = "DEBUG"
            mock_settings.service_name = "model-registry"
            mock_settings.environment = "test"

            with patch("logging_config.get_settings", return_value=mock_settings):
                log_module.configure_logging()

            assert log_module._configured is True
        finally:
            log_module._configured = False

    def test_configure_logging_console_format(self) -> None:
        """configure_logging with console format configures structlog."""
        log_module._configured = False
        try:
            mock_settings = MagicMock()
            mock_settings.log_format = "console"
            mock_settings.log_level = "INFO"
            mock_settings.service_name = "model-registry"
            mock_settings.environment = "test"

            with patch("logging_config.get_settings", return_value=mock_settings):
                log_module.configure_logging()

            assert log_module._configured is True
        finally:
            log_module._configured = False

    def test_configure_logging_idempotent(self) -> None:
        """Calling configure_logging multiple times is safe."""
        log_module._configured = False
        try:
            log_module.configure_logging()
            # Second call should be a no-op
            log_module.configure_logging()
            assert log_module._configured is True
        finally:
            log_module._configured = False


class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_default_name_can_log(self) -> None:
        """get_logger with no name returns a logger that can emit structured messages."""
        log_module._configured = False
        try:
            logger = log_module.get_logger()
            # Actually invoke the logging method to verify it works end-to-end
            # (structlog loggers are callable proxies; calling .info() should not raise)
            logger.info("test_event", key="value")
            # Verify it's a structlog BoundLogger (not a stdlib logger)
            assert type(logger).__name__ in ("BoundLogger", "BoundLoggerLazyProxy")
        finally:
            log_module._configured = False

    def test_get_logger_custom_name_can_log(self) -> None:
        """get_logger with a custom name returns a logger bound to that module."""
        log_module._configured = False
        try:
            logger = log_module.get_logger("my_module")
            # Actually emit a structured log message to verify it works
            logger.info("custom_module_event", module="my_module")
            assert type(logger).__name__ in ("BoundLogger", "BoundLoggerLazyProxy")
        finally:
            log_module._configured = False

    def test_get_logger_configures_logging(self) -> None:
        """get_logger calls configure_logging if not already configured."""
        log_module._configured = False
        try:
            assert log_module._configured is False
            log_module.get_logger("test_auto_configure")
            assert log_module._configured is True
        finally:
            log_module._configured = False
