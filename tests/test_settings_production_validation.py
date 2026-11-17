"""Settings refuse to start in production with empty URL placeholders."""

import pytest

from settings import Settings


def _mk(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@h:5432/db",  # pragma: allowlist secret
        "minio_endpoint": "minio:9000",
        "otlp_traces_endpoint": "http://alloy:4318/v1/traces",
        "rtx_endpoint": "http://v:8000",
        "adapter_manager_url": "http://v:8001",
        "environment": "production",
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestProductionValidation:
    def test_complete_config_allows_init(self) -> None:
        settings = _mk()
        assert settings.environment == "production"

    @pytest.mark.parametrize(
        "empty_field",
        [
            "database_url",
            "minio_endpoint",
            "otlp_traces_endpoint",
            "rtx_endpoint",
        ],
    )
    def test_empty_required_raises(self, empty_field: str) -> None:
        with pytest.raises(RuntimeError, match=empty_field):
            _mk(**{empty_field: ""})

    def test_empty_adapter_manager_url_is_allowed(self) -> None:
        """adapter_manager_url is NOT a startup requirement.

        adapter-manager is deployed nowhere, so a startup guard demanding its
        URL can never be satisfied. It is enforced at the point of use instead
        (DeployService._load_adapter raises DeployError).
        """
        settings = _mk(adapter_manager_url="")
        assert settings.adapter_manager_url == ""

    def test_test_environment_allows_empty(self) -> None:
        Settings.model_validate({"environment": "test"})  # must not raise

    def test_non_production_environment_does_not_validate(self) -> None:
        """The guard is keyed on the literal string 'production'.

        This pins a KNOWN GAP rather than desired behaviour: the live
        Deployment sets MODEL_REGISTRY_ENVIRONMENT=strato, so the guard never
        fires in the real production pod. It is not widened here because
        otlp_traces_endpoint is empty in that pod and widening would
        crash-loop it. See the comment on Settings._REQUIRED_IN_PRODUCTION.
        """
        settings = Settings.model_validate({"environment": "strato"})
        assert settings.rtx_endpoint == ""
