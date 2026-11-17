"""Application settings for model-registry service.

Uses Pydantic BaseSettings for type-safe configuration from environment variables.
All settings are prefixed with MODEL_REGISTRY_ in the environment.
"""

import tempfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_package_version() -> str:
    """Read version from package metadata, falling back to '0.0.0-dev'.

    Only catches the documented failure modes of ``importlib.metadata.version``:
    a missing distribution (``PackageNotFoundError``) or a missing
    ``importlib.metadata`` backport on exotic Python builds (``ImportError``).
    Any other failure indicates a real bug and should propagate.
    """
    try:
        return _pkg_version("model-registry")
    except PackageNotFoundError, ImportError:
        return "0.0.0-dev"


# ---------------------------------------------------------------------------
# Defaults -- used only when a caller omits these fields on model registration.
# Concrete model + quant identifiers belong in gitops (callers supply them
# explicitly via the training pipeline); these placeholders just keep the
# ORM columns non-null for legacy rows.
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "unspecified"
DEFAULT_QUANTIZATION = "unspecified"

# Server-side default provider chain, applied when a caller creates a routing
# rule without naming one.
#
# ⚠️ 2026-08-17 -- this was ``["deepseek", "claude"]`` and had been WRONG since
# ``claude`` was disabled in inference-router's provider catalog. ``claude`` is
# ``enabled: false`` there, so inference-router's ``_build_provider_chain``
# filtered it out and every rule created from this default collapsed to the
# single hop ``["deepseek"]`` -- i.e. the server's OWN default produced a rule
# with no failover, and did so silently. It now matches the live
# ``default_chain`` in inference-router's providers ConfigMap
# (``[pgx-local, deepseek]``) and the 99 correctly-chained production rules.
#
# This value must satisfy ``validate_fallback_chain`` (see
# ``services/chain_validation``) -- a default that its own guard rejects is a
# startup-time contradiction, and ``tests/test_chain_validation.py`` pins that.
DEFAULT_FALLBACK_CHAIN: list[str] = ["pgx-local", "deepseek"]


class Settings(BaseSettings):
    """Model registry application settings.

    All environment variables should be prefixed with MODEL_REGISTRY_.
    For example: MODEL_REGISTRY_PORT=8155
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_REGISTRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Required by ``otlp_traces_endpoint``'s ``validation_alias`` below.
        # Setting a validation_alias makes pydantic accept ONLY the alias, so
        # without this every `Settings.model_validate({"otlp_traces_endpoint":
        # ...})` in the test-suite silently stops populating the field. Six
        # tests in tests/test_settings_production_validation.py caught exactly
        # that. populate_by_name restores the field name as an accepted key
        # while keeping both env aliases working.
        populate_by_name=True,
    )

    # Service metadata
    service_name: str = Field(default="model-registry", description="Service name for telemetry")
    port: int = Field(default=8155, description="HTTP port to listen on")
    version: str = Field(default_factory=_get_package_version, description="Service version")

    # Database -- DSN comes from env (MODEL_REGISTRY_DATABASE_URL).
    database_url: str = Field(default="", description="PostgreSQL DSN (psycopg3 driver)")

    # Routing cache
    routing_cache_ttl: int = Field(
        default=60, description="TTL in seconds for routing table cache entries"
    )

    # ---------------------------------------------------------------------
    # Provider catalog -- the allowlist that makes an unroutable fallback
    # chain impossible to REGISTER (see services/chain_validation.py).
    #
    # Why this lives here and not only in inference-router: IR owns the
    # runtime catalog (its providers ConfigMap) and filters unknown/disabled
    # names out of a chain at request time with nothing but a WARNING. That
    # filter is invisible to whoever wrote the rule, so between 2026-05-10
    # and 2026-08-16, 64 of 163 production rules accumulated a chain that
    # could not fail over at all:
    #   * 61 rules  ["pgx-local"]                 -- frozen before deepseek
    #                                                was enabled (2026-06-16)
    #   *  3 rules  ["gemma","deepseek","claude"] -- `gemma` is not a
    #                                                provider name and
    #                                                `claude` is disabled, so
    #                                                the chain collapsed to
    #                                                one hop
    # model-registry is the ONLY chokepoint both writers pass through
    # (IR's POST /api/routing/ensure and coding-client's
    # PUT /api/routing/{schema_id}), so the allowlist has to be readable
    # here to reject the write rather than report the damage later.
    #
    # Format: comma-separated ``name=true|false`` mirroring the
    # ``enabled:`` flag of each entry in IR's providers.yaml. Render both
    # from the same gitops value. Drift is LOUD, not silent: a provider
    # added to IR but not here is rejected with a 422 naming it, which is
    # the safe direction -- the unsafe direction (silently accepting a name
    # nothing can route) is exactly the defect above.
    routing_provider_catalog: str = Field(
        default="pgx-local=true,deepseek=true,xai=true,claude=false",
        description=(
            "Provider allowlist for routing-rule fallback chains, as "
            "'name=true|false' pairs mirroring inference-router's provider "
            "catalog (true = enabled). Chains naming a provider absent from "
            "this list, or with fewer than routing_min_viable_hops distinct "
            "ENABLED providers, are rejected at registration."
        ),
    )
    routing_min_viable_hops: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum number of distinct ENABLED providers a fallback chain "
            "must name. 2 = 'must be able to fail over at least once'. "
            "Set to 1 only to deliberately permit single-provider "
            "(no-failover) rules fleet-wide."
        ),
    )

    # ---------------------------------------------------------------------
    # Chain reconciliation -- the DETECTIVE half of the same defect.
    #
    # The catalog above lets the registration guard reject a bad chain at
    # write time. It cannot see a chain that was valid when written and
    # rotted later because a provider was flipped to ``enabled: false`` in
    # inference-router -- no write happens, so no guard runs. That is exactly
    # how the three ["gemma","deepseek","claude"] rules died. The sweep in
    # services/chain_reconciliation.py re-checks every STORED rule against
    # the current catalog and COUNTS the unroutable ones. It never repairs
    # them and never fails closed.
    routing_reconcile_interval_seconds: int = Field(
        default=3600,
        ge=0,
        description=(
            "How often to re-check stored fallback chains against the "
            "provider catalog. A sweep also runs once at startup, so a "
            "provider flip is caught on the next roll. 0 disables the "
            "background loop entirely (the on-demand endpoint still works) "
            "and logs a warning saying so -- silence is not an option."
        ),
    )
    routing_reconcile_report_limit: int = Field(
        default=50,
        ge=1,
        description=(
            "Cap on how many individual unroutable rules a report ENUMERATES. "
            "The COUNTS are always exact and unaffected; only the per-rule "
            "list is capped, with findings_truncated set, so one bad catalog "
            "change cannot emit a 163-entry log line or response body."
        ),
    )

    # MinIO / object storage (used by model-trainer for uploads; registry stores metadata only)
    minio_endpoint: str = Field(default="", description="MinIO endpoint for GGUF file storage")
    minio_access_key: str = Field(default="", description="MinIO access key")
    minio_secret_key: str = Field(default="", description="MinIO secret key")

    # GitOps deployment
    gitops_repo_path: str = Field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "gitops"),
        description="Local path to a GitOps repository clone for deployment triggers",
    )
    gitops_enabled: bool = Field(
        default=False,
        description="Enable actual git commit/push after writing values YAML. Opt-in; disabled by default.",
    )

    # Observability
    #
    # ⚠️ Read the FLEET-WIDE name first. Every service's Deployment sets the
    # unprefixed ``OTLP_TRACES_ENDPOINT``, but this class carries
    # ``env_prefix="MODEL_REGISTRY_"``, and an env_prefix does NOT extend to a
    # name the manifest already spells without it. So this field bound nothing,
    # resolved to "", and ``telemetry.py`` fell back to host "alloy", failed its
    # reachability probe, and disabled tracing with a warning nobody read.
    #
    # ``validation_alias`` overrides the prefix entirely, so the prefixed name
    # is listed explicitly as a second choice rather than being lost. First
    # match wins, so an explicit MODEL_REGISTRY_ override still takes priority
    # if someone sets it.
    #
    # This is the same defect and the same fix as inference-router#98, which is
    # verified live: that pod now logs "OpenTelemetry instrumentation enabled"
    # and its spans reach tempo. Fix the CODE to the fleet name — do not change
    # the manifest to match one service.
    otlp_traces_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MODEL_REGISTRY_OTLP_TRACES_ENDPOINT",
            "OTLP_TRACES_ENDPOINT",
        ),
        description="OTLP HTTP endpoint for trace export",
    )

    # Runtime
    environment: str = Field(default="production", description="Deployment environment")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json", description="Log format: 'json' for production, 'console' for development"
    )

    # Auto-approval
    auto_approve_score: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum eval_score for automatic approval when transitioning to 'evaluated'. "
        "Set to 1.0 to effectively disable auto-approval.",
    )

    # Deployment tracking
    deployment_tracking_enabled: bool = Field(
        default=False,
        description="Enable deploy_status tracking fields on model versions. When disabled, existing fire-and-forget behavior is preserved.",
    )

    # vLLM server / adapter manager. The field name rtx_endpoint is kept for
    # config back-compat so the existing MODEL_REGISTRY_RTX_ENDPOINT env var binds.
    rtx_endpoint: str = Field(default="", description="Shared vLLM endpoint")
    adapter_manager_url: str = Field(default="", description="Adapter manager sidecar URL")

    # Cluster DNS suffix + listen port for fg-* deployed model services.
    # Externalized so a non-default cluster domain works without code changes; the
    # values written into Helm values YAML by DeployService._compute_endpoint()
    # come from here. The default is an example domain, never a real cluster.
    cluster_dns_domain: str = Field(
        default="example.com",
        description="In-cluster DNS suffix used to build deployed-model endpoints",
    )
    deployed_service_port: int = Field(
        default=8080,
        description="Listen port of fg-* deployed model services",
    )

    # NOTE (2026-07-19): this guard only fires when ``environment`` is the
    # literal string "production". The live Deployment sets
    # MODEL_REGISTRY_ENVIRONMENT=strato, so it is currently INERT in the real
    # production pod.
    #
    # UPDATE (2026-08-09): the reason it could not safely be widened is GONE.
    # It used to be that ``otlp_traces_endpoint`` was empty in the live pod --
    # the Deployment sets the unprefixed OTLP_TRACES_ENDPOINT, which did not
    # bind through the MODEL_REGISTRY_ prefix -- so widening this guard would
    # have crash-looped the pod on startup. The field now reads the unprefixed
    # fleet name via ``validation_alias`` (see above), so it binds from the
    # existing manifest with no gitops change. Widening the guard to a
    # fail-closed allowlist is therefore now a safe, separate change; it is left
    # out of this one only to keep a telemetry fix from also altering startup
    # failure semantics.
    #
    # ``adapter_manager_url`` is not required: an adapter manager is an optional
    # external sidecar, so demanding its URL at startup is a requirement nothing
    # can satisfy. It is instead enforced at the point of use — DeployService._load_adapter
    # raises DeployError when it is unset, which is the only code path that needs it.
    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "database_url",
        "minio_endpoint",
        "otlp_traces_endpoint",
        "rtx_endpoint",
    )

    def model_post_init(self, __context: object) -> None:
        """Normalize database URL + validate required fields in production."""
        if self.database_url.startswith("postgresql+asyncpg://"):
            self.database_url = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql+psycopg://", 1
            )
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        if self.environment == "production":
            missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
            if missing:
                raise RuntimeError(
                    f"model-registry settings missing required fields in production: {missing}. "
                    "Set them via environment variables."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Returns:
        The application Settings parsed from environment variables.
    """
    return Settings()
