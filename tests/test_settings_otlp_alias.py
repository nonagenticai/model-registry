"""The OTLP endpoint must bind from the FLEET-WIDE unprefixed env var.

Regression guard for 2026-08-09. Every service's Deployment sets the unprefixed
``OTLP_TRACES_ENDPOINT``, but ``Settings`` carries ``env_prefix="MODEL_REGISTRY_"``
and an env_prefix does NOT extend to a name the manifest already spells without
it. So the field bound nothing, resolved to ``""``, and ``telemetry.py`` fell
back to host ``"alloy"``, failed its reachability probe, and disabled tracing
with a warning nobody read -- while every other observability signal looked fine.

Same defect and same fix as inference-router#98, which is verified live.

Each test below fails against the pre-fix ``Field(default="")`` with no
``validation_alias``, except ``test_field_name_still_populates``, which fails
against a naive fix that adds the alias WITHOUT ``populate_by_name=True`` --
that variant silently stops accepting the field name and takes six existing
tests in ``test_settings_production_validation.py`` down with it.
"""

import pytest

from settings import Settings

_FLEET = "http://tempo.tempo.svc.example.com:4318/v1/traces"
_EXPLICIT = "http://explicit-override:4318/v1/traces"


@pytest.fixture(autouse=True)
def _clean_otlp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip both spellings so a developer's shell cannot mask a regression."""
    monkeypatch.delenv("OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_OTLP_TRACES_ENDPOINT", raising=False)
    # Keep the production guard out of the way -- this module is about binding,
    # not about the startup allowlist. This is what the live pod actually sets.
    monkeypatch.setenv("MODEL_REGISTRY_ENVIRONMENT", "strato")


def test_unprefixed_fleet_name_binds() -> None:
    """THE bug. The live Deployment sets exactly this, and it used to be ignored."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OTLP_TRACES_ENDPOINT", _FLEET)
        assert Settings().otlp_traces_endpoint == _FLEET


def test_prefixed_name_still_binds() -> None:
    """The documented MODEL_REGISTRY_ convention must not regress."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_REGISTRY_OTLP_TRACES_ENDPOINT", _EXPLICIT)
        assert Settings().otlp_traces_endpoint == _EXPLICIT


def test_prefixed_name_wins_over_fleet_name() -> None:
    """Precedence is deliberate: an explicit per-service override beats the default."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OTLP_TRACES_ENDPOINT", _FLEET)
        mp.setenv("MODEL_REGISTRY_OTLP_TRACES_ENDPOINT", _EXPLICIT)
        assert Settings().otlp_traces_endpoint == _EXPLICIT


def test_absent_from_both_is_empty_not_an_error() -> None:
    """Tracing stays opt-in. Neither name set must not raise outside production."""
    assert Settings().otlp_traces_endpoint == ""


def test_field_name_still_populates() -> None:
    """``validation_alias`` alone would break every model_validate() call site.

    Setting a validation_alias makes pydantic accept ONLY the alias unless
    ``populate_by_name=True`` is also set. Without it this assertion fails, and
    so do six tests in test_settings_production_validation.py which all build a
    Settings from a dict keyed by field name.
    """
    settings = Settings.model_validate(
        {
            "database_url": "postgresql+psycopg://u:p@h:5432/db",  # pragma: allowlist secret
            "minio_endpoint": "minio:9000",
            "otlp_traces_endpoint": _FLEET,
            "rtx_endpoint": "http://v:8000",
            "environment": "production",
        }
    )
    assert settings.otlp_traces_endpoint == _FLEET
