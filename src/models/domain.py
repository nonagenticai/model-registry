"""Pydantic v2 domain models for the model registry.

These models represent the core business objects: model versions, routing rules,
and the API request/response schemas that wrap them.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field

from settings import DEFAULT_BASE_MODEL, DEFAULT_FALLBACK_CHAIN, DEFAULT_QUANTIZATION

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AuditEventType(StrEnum):
    """Types of auditable events in the model registry."""

    MODEL_REGISTERED = "model_registered"
    STATUS_CHANGED = "status_changed"
    ROUTE_UPDATED = "route_updated"
    ROUTE_DELETED = "route_deleted"
    ALIAS_SET = "alias_set"
    ALIAS_DELETED = "alias_deleted"
    TAG_SET = "tag_set"
    TAG_DELETED = "tag_deleted"
    DEPLOYMENT_TRIGGERED = "deployment_triggered"
    AUTO_APPROVED = "auto_approved"
    GLOBAL_ADAPTER_SET = "global_adapter_set"


class DeployStatus(StrEnum):
    """Deployment pipeline status for a model version."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelStatus(StrEnum):
    """Lifecycle status of a model version.

    Valid transitions:
        training -> evaluating
        evaluating -> evaluated
        evaluated -> approved
        approved -> exporting
        exporting -> deployed
        deployed -> retired
        Any -> retired  (cleanup escape hatch)
    """

    TRAINING = "training"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    EXPORTING = "exporting"
    DEPLOYED = "deployed"
    RETIRED = "retired"


# Mapping of valid state transitions: current_status -> set of allowed next statuses
VALID_TRANSITIONS: dict[ModelStatus, set[ModelStatus]] = {
    ModelStatus.TRAINING: {ModelStatus.EVALUATING, ModelStatus.RETIRED},
    ModelStatus.EVALUATING: {ModelStatus.EVALUATED, ModelStatus.RETIRED},
    ModelStatus.EVALUATED: {ModelStatus.APPROVED, ModelStatus.RETIRED},
    ModelStatus.APPROVED: {ModelStatus.EXPORTING, ModelStatus.RETIRED},
    # EXPORTING -> APPROVED is the failed-deploy path: DeployService rolls a
    # model back to approved when placement fails, so the deploy is retryable.
    # Without it, exporting's only exit is RETIRED while POST /api/deploy
    # requires APPROVED — a failed deploy would strand the model permanently.
    ModelStatus.EXPORTING: {ModelStatus.DEPLOYED, ModelStatus.APPROVED, ModelStatus.RETIRED},
    ModelStatus.DEPLOYED: {ModelStatus.RETIRED},
    ModelStatus.RETIRED: set(),  # Terminal state
}


def is_valid_transition(current: ModelStatus, target: ModelStatus) -> bool:
    """Check whether a status transition is allowed.

    Args:
        current: The current model status.
        target: The desired new status.

    Returns:
        True if the transition is valid, False otherwise.
    """
    allowed = VALID_TRANSITIONS.get(current, set())
    return target in allowed


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class ModelVersion(BaseModel):
    """A registered model version in the catalog.

    Tracks a specific fine-tuned Qwen3-4B-AWQ model through its lifecycle
    from training to deployment on the Pi inference cluster.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid7, description="Unique model version identifier")
    schema_id: str = Field(
        ...,
        description=(
            "Schema identifier. For routes auto-registered from an OpenAI "
            "`response_format` this is the uuid7 allocated by model-registry "
            "on first sighting; opaque to callers."
        ),
    )
    version: str = Field(..., description="Version label (e.g. 'v3')")
    status: ModelStatus = Field(
        default=ModelStatus.TRAINING, description="Current lifecycle status"
    )
    gguf_file: str | None = Field(default=None, description="MinIO path to the GGUF model file")
    gguf_sha256: str | None = Field(default=None, description="SHA256 checksum of the GGUF file")
    adapter_path: str | None = Field(
        default=None, description="MinIO path to PEFT adapter directory"
    )
    model_format: Literal["gguf", "peft"] = Field(
        default="gguf", description="Model artifact format"
    )
    base_model: str = Field(
        default=DEFAULT_BASE_MODEL,
        description=(
            "Base model identifier the training run started from. Supplied by "
            "the trainer; concrete model names live in gitops manifests, not here."
        ),
    )
    quantization: str = Field(
        default=DEFAULT_QUANTIZATION,
        description="Quantization format (e.g. 'awq_marlin', 'gptq_marlin', 'nvfp4', 'Q4_K_M')",
    )
    training_samples: int = Field(default=0, ge=0, description="Number of training samples used")
    eval_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Composite evaluation score (0.0 to 1.0)"
    )
    eval_metrics: dict[str, Any] | None = Field(
        default=None, description="Detailed evaluation metrics dictionary"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this version was registered"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last modification timestamp"
    )
    deployed_at: datetime | None = Field(
        default=None, description="When this version was deployed to production"
    )
    deploy_status: DeployStatus | None = Field(
        default=None, description="Current deployment pipeline status"
    )
    deploy_error: str | None = Field(default=None, description="Error message if deployment failed")
    deploy_started_at: datetime | None = Field(
        default=None, description="When deployment was initiated"
    )
    deploy_completed_at: datetime | None = Field(
        default=None, description="When deployment completed (success or failure)"
    )


class RoutingRule(BaseModel):
    """A routing rule mapping a schema_id to its active model and fallback chain.

    The inference router uses these rules to decide where to send each request.
    """

    model_config = ConfigDict(from_attributes=True)

    schema_id: str = Field(..., description="Schema identifier that this rule routes for")
    schema_fingerprint: str | None = Field(
        default=None,
        description=(
            "Content hash of the response_format JSON schema. Set for "
            "auto-registered rules; NULL for legacy rules that were PUT "
            "with a caller-chosen schema_id."
        ),
    )
    active_model_id: UUID | None = Field(
        default=None, description="UUID of the currently active model version"
    )
    fallback_chain: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FALLBACK_CHAIN),
        description="Ordered fallback inference backends",
    )
    endpoint: str | None = Field(
        default=None,
        description=(
            "Optional direct-endpoint override. When null/empty the rule routes "
            "through the provider chain (inference-router provider catalog); "
            "set only for bespoke per-schema backends."
        ),
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="Output JSON schema for response validation"
    )
    gbnf_grammar: str | None = Field(
        default=None, description="GBNF grammar for constrained decoding on llama.cpp"
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt injected at inference time. Used by training-data-generator "
        "to include the same system message in training data.",
    )
    adapter_path: str | None = Field(default=None, description="MinIO path to active PEFT adapter")
    adapter_loaded: bool = Field(
        default=False,
        description="Whether the adapter is currently loaded into vLLM",
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Whether this per-schema rule is active. When false the "
            "inference-router falls through to the global adapter. Defaults "
            "to true so existing rules retain current behaviour."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this rule was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last modification timestamp"
    )


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class RegisterModelRequest(BaseModel):
    """Request body for POST /api/models."""

    schema_id: str = Field(..., description="Schema identifier for this model")
    version: str = Field(..., description="Version label")
    base_model: str = Field(default=DEFAULT_BASE_MODEL, description="Base model name")
    quantization: str = Field(default=DEFAULT_QUANTIZATION, description="Quantization format")
    training_samples: int = Field(default=0, ge=0, description="Number of training samples")
    gguf_file: str | None = Field(default=None, description="MinIO path to GGUF file")
    gguf_sha256: str | None = Field(default=None, description="SHA256 of GGUF file")
    adapter_path: str | None = Field(
        default=None, description="MinIO path to PEFT adapter directory"
    )
    model_format: str = Field(default="gguf", description="Model format: 'gguf' or 'peft'")
    status: ModelStatus = Field(
        default=ModelStatus.TRAINING, description="Initial lifecycle status"
    )
    tags: dict[str, str] | None = Field(
        default=None, description="Optional key-value tags for lineage tracking"
    )


class UpdateStatusRequest(BaseModel):
    """Request body for PATCH /api/models/{model_id}/status."""

    status: ModelStatus = Field(..., description="Target lifecycle status")
    eval_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Evaluation score (set when moving to evaluated)"
    )
    eval_metrics: dict[str, Any] | None = Field(
        default=None, description="Evaluation metrics (set when moving to evaluated)"
    )
    gguf_file: str | None = Field(default=None, description="GGUF path (set when exporting)")
    gguf_sha256: str | None = Field(default=None, description="GGUF checksum (set when exporting)")


class UpdateRoutingRequest(BaseModel):
    """Request body for PUT /api/routing/{schema_id}."""

    active_model_id: UUID | None = Field(default=None, description="Active model version UUID")
    fallback_chain: list[str] | None = Field(default=None, description="Ordered fallback backends")
    endpoint: str | None = Field(default=None, description="In-cluster endpoint URL")
    json_schema: dict[str, Any] | None = Field(default=None, description="Output JSON schema")
    gbnf_grammar: str | None = Field(default=None, description="GBNF grammar string")
    system_prompt: str | None = Field(
        default=None, description="System prompt for inference and training data"
    )
    adapter_path: str | None = Field(default=None, description="MinIO path to PEFT adapter")
    enabled: bool | None = Field(
        default=None,
        description=(
            "Optional. Set false to disable a per-schema rule (inference-router "
            "will fall through to the global adapter). Omitted = leave unchanged "
            "on update; new rules default to true."
        ),
    )


class EnsureRoutingRequest(BaseModel):
    """Request body for POST /api/routing/ensure.

    Upserts a routing rule keyed by ``fingerprint``. If a rule with this
    fingerprint already exists, it is returned unchanged. Otherwise a new
    rule is created with a server-generated uuid7 ``schema_id``.

    Lets callers register schemas by content rather than by a chosen id —
    the inference-router derives the fingerprint from the caller's
    ``response_format`` and lets model-registry allocate the internal id.
    """

    fingerprint: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Content hash of the canonical JSON schema",
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="Output JSON schema for validation + constrained decoding"
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt to attach to the rule"
    )
    fallback_chain: list[str] | None = Field(
        default=None, description="Provider fallback chain (uses server default if omitted)"
    )


class DeployRequest(BaseModel):
    """Request body for POST /api/deploy/{model_id}."""

    target_namespace: str = Field(
        default="llm-inference", description="Kubernetes namespace for the deployment"
    )
    replicas: int = Field(default=1, ge=1, le=10, description="Number of pod replicas")


class DeployResponse(BaseModel):
    """Response body for POST /api/deploy/{model_id}."""

    model_id: UUID
    schema_id: str
    status: ModelStatus
    values_file: str | None = None
    message: str


class ModelListResponse(BaseModel):
    """Response body for GET /api/models."""

    models: list[ModelVersion]
    total: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "healthy"
    service: str = "model-registry"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LivenessResponse(BaseModel):
    """Response body for GET /health/live."""

    status: str = "alive"


class ReadinessResponse(BaseModel):
    """Response body for GET /health/ready."""

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class QualityResponse(BaseModel):
    """Response body for GET /api/models/{schema_id}/quality.

    Returns the latest quality score for a schema, derived from the most
    recent model version that has a non-null eval_score.
    """

    schema_id: str
    quality_score: float | None = None
    model_id: UUID | None = None
    version: str | None = None


class UnroutableRuleReport(BaseModel):
    """One stored routing rule the current provider catalog cannot route."""

    model_config = ConfigDict(from_attributes=True)

    schema_id: str = Field(description="Primary key of the offending rule")
    fallback_chain: list[str] = Field(
        description=(
            "The chain AS STORED, never a repaired version -- the report "
            "exists to show what is actually in the table"
        )
    )
    enabled: bool = Field(
        description=(
            "The rule's own enabled flag. False means it is harmless today "
            "but is a revival hazard: re-enabling it does not re-validate "
            "the chain"
        )
    )
    reason: str = Field(
        description="Why it cannot fail over, in the registration guard's own words"
    )


class ChainHealthResponse(BaseModel):
    """Response body for GET /api/routing/chain-health.

    Reports whether stored fallback chains are still routable against the
    CURRENT provider catalog. Purely descriptive: requesting it never
    repairs a rule and never changes what any other endpoint returns.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(
        description=(
            "'ok', 'unroutable_rules_found', or 'unknown'. 'unknown' means "
            "the catalog could not be read so NOTHING was checked -- it is a "
            "distinct value so it can never be read as 'ok'"
        )
    )
    scanned: int = Field(description="Rules examined; 0 when status is 'unknown'")
    unroutable: int | None = Field(
        default=None,
        description=(
            "Total unroutable rules, or null when status is 'unknown'. Null "
            "rather than 0: 'could not check' must not render as 'clean'"
        ),
    )
    unroutable_enabled: int | None = Field(
        default=None,
        description="Unroutable rules that are ENABLED -- the count worth alerting on",
    )
    unroutable_disabled: int | None = Field(
        default=None,
        description="Unroutable rules that are switched off -- the revival hazard",
    )
    catalog: dict[str, bool] | None = Field(
        default=None,
        description=(
            "The provider catalog the sweep resolved, so a reader can tell a "
            "real regression from a catalog that changed underneath it"
        ),
    )
    detail: str = Field(description="One actionable sentence; spells out any 'unknown' cause")
    completed_at: datetime = Field(
        description=(
            "When the sweep finished. A stale timestamp means the reconciler "
            "stopped running, which is invisible without this field"
        )
    )
    findings: list[UnroutableRuleReport] = Field(
        default_factory=list, description="The offending rules, capped by report_limit"
    )
    findings_truncated: bool = Field(
        default=False,
        description=(
            "True when 'findings' was capped, so the list length is never mistaken for the count"
        ),
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str


# ---------------------------------------------------------------------------
# Audit event models
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    """A single audit event recording a mutation in the registry."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Auto-incremented event ID")
    timestamp: datetime = Field(description="When the event occurred")
    event_type: AuditEventType = Field(description="Type of event")
    entity_type: str = Field(
        description="Type of entity affected (model_version, routing_rule, alias, tag)"
    )
    entity_id: str = Field(description="Identifier of the affected entity")
    actor: str = Field(default="system", description="Who/what triggered the event")
    detail: dict[str, Any] | None = Field(
        default=None, description="Additional event details as JSON"
    )


class AuditEventListResponse(BaseModel):
    """Response body for audit event listing endpoints."""

    events: list[AuditEvent]
    total: int


# ---------------------------------------------------------------------------
# Alias models
# ---------------------------------------------------------------------------


class ModelAlias(BaseModel):
    """A named alias pointing to a specific model version within a schema."""

    model_config = ConfigDict(from_attributes=True)

    schema_id: str = Field(description="Schema identifier")
    alias_name: str = Field(description="Alias name (e.g. 'champion', 'rollback')")
    model_id: UUID = Field(description="UUID of the model version this alias points to")
    created_at: datetime = Field(description="When this alias was first created")
    updated_at: datetime = Field(description="When this alias was last updated")


class SetAliasRequest(BaseModel):
    """Request body for PUT /api/aliases/{schema_id}/{alias_name}."""

    model_id: UUID = Field(description="UUID of the model version to alias")


class AliasListResponse(BaseModel):
    """Response body for alias listing endpoints."""

    aliases: list[ModelAlias]


# ---------------------------------------------------------------------------
# Tag models
# ---------------------------------------------------------------------------


class ModelTag(BaseModel):
    """A key-value tag attached to a model version."""

    model_config = ConfigDict(from_attributes=True)

    model_id: UUID = Field(description="UUID of the model version")
    key: str = Field(description="Tag key")
    value: str = Field(description="Tag value")
    created_at: datetime = Field(description="When this tag was created")


class SetTagsRequest(BaseModel):
    """Request body for PUT /api/models/{model_id}/tags."""

    tags: dict[str, str] = Field(description="Key-value pairs to set as tags")


class TagListResponse(BaseModel):
    """Response body for tag listing endpoints."""

    tags: list[ModelTag]


# ---------------------------------------------------------------------------
# Model comparison models
# ---------------------------------------------------------------------------


class ModelVersionSummary(BaseModel):
    """Summary of a model version for comparison, including aliases and tags."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    schema_id: str
    version: str
    status: ModelStatus
    eval_score: float | None = None
    eval_metrics: dict[str, Any] | None = None
    created_at: datetime
    deployed_at: datetime | None = None
    deploy_status: DeployStatus | None = None
    aliases: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)


class ModelComparison(BaseModel):
    """Response body for model version comparison."""

    schema_id: str
    versions: list[ModelVersionSummary]


# ---------------------------------------------------------------------------
# Deployment feedback models
# ---------------------------------------------------------------------------


class DeployFeedbackRequest(BaseModel):
    """Request body for PATCH /api/models/{model_id}/deploy-status."""

    deploy_status: DeployStatus = Field(description="Deployment status feedback")
    deploy_error: str | None = Field(default=None, description="Error message if deployment failed")


# ---------------------------------------------------------------------------
# Global adapter models (single-LoRA rollout)
# ---------------------------------------------------------------------------


class GlobalAdapter(BaseModel):
    """The system-wide default LoRA adapter.

    Single-row substrate populated by model-trainer after a successful
    ``mode=global`` run; consumed by adapter-manager + inference-router as
    the fall-through adapter when no enabled per-schema rule matches.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(default=1, description="Singleton id (always 1)")
    model_version_id: UUID = Field(
        ..., description="UUID of the model_version row this adapter belongs to"
    )
    minio_uri: str = Field(..., description="MinIO URI of the LoRA adapter directory")
    adapter_hash: str = Field(
        ..., description="Content hash of the adapter (used for cache invalidation)"
    )
    notes: str | None = Field(
        default=None, description="Optional notes (e.g. training-run summary)"
    )
    created_at: datetime = Field(description="When the global adapter was first set")
    updated_at: datetime = Field(description="Last upsert timestamp")


class UpsertGlobalAdapterRequest(BaseModel):
    """Request body for POST /api/global-adapter."""

    model_version_id: UUID = Field(
        ..., description="UUID of the model_version row this adapter belongs to"
    )
    minio_uri: str = Field(
        ..., min_length=1, max_length=1024, description="MinIO URI of the LoRA adapter directory"
    )
    adapter_hash: str = Field(
        ..., min_length=1, max_length=128, description="Content hash of the adapter"
    )
    notes: str | None = Field(default=None, description="Optional notes")
