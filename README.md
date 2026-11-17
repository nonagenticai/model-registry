# Model Registry

Central catalog of model versions and routing table for an inference cluster. Port 8155.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · SQLAlchemy · Pydantic |
| **Database** | PostgreSQL 18 |

## What Model Registry Does

When a training run completes, the trainer registers the new model version here by POSTing its
schema ID, version label, base model, quantization format, eval score, and storage path. The
registry assigns a UUID, sets the initial lifecycle status to "training", and persists the
record in PostgreSQL. From that point on, every status transition — training, evaluating,
evaluated, approved, exporting, deployed, retired — is tracked as an audit event, giving the
platform a complete history of every model that has ever existed.

The registry does more than store metadata. It maintains a routing table that maps each schema
ID to an active model endpoint and a fallback provider chain. It also owns the
**`global_adapter`** row that names the system-wide single LoRA adapter that an adapter manager
hot-loads into vLLM; per-schema rules are kept with an `enabled` flag and are re-enabled per
schema only when the global adapter underperforms a specific case. When an inference router
receives a request for a schema, it queries this routing table to decide which endpoint to hit
and what fallback chain to use. The routing table is served from an in-memory TTL cache
(default 60 seconds) so lookups add negligible latency. On deployment, the deploy endpoint
transitions the model through exporting to deployed, generates Helm values YAML for a GitOps
reconciler (GGUF models) or upserts the PEFT adapter's reference into the routing/global-adapter
tables (adapter-manager picks it up and hot-loads it into vLLM), updates the routing table with
the new endpoint, and rotates the "champion" alias while preserving the previous champion as
"rollback".

Named aliases decouple model identity from version numbers. A router or the training loop can
ask for "champion" without knowing the specific UUID, and the registry resolves it. This enables
zero-downtime rollback: if a newly deployed model underperforms, switching the champion alias
back to rollback instantly reroutes traffic without redeploying anything. Every mutation — status
change, route update, alias rotation, tag edit — is recorded in the audit log with timestamp,
actor, and detail payload.

The service is stateless in compute but stateful in data. It owns no inference logic and holds
no model weights — it is purely a metadata and routing coordination layer. All data lives in
PostgreSQL.

## Features

- **Model version lifecycle** — tracks each model from training through evaluation, approval,
  deployment, and retirement with enforced state-machine transitions
- **Routing table with TTL cache** — maps schema IDs to active endpoints and fallback chains,
  served from memory so the router gets sub-millisecond lookups
- **Dual deployment paths** — GGUF models deploy via Helm values YAML for a GitOps reconciler;
  PEFT models are recorded for an adapter manager to hot-load
- **Named aliases** — champion, rollback, and custom pointers decouple model identity from
  version UUIDs
- **Auto-approval** — models whose eval score exceeds a configurable threshold (default 0.95)
- **Audit trail** — every mutation recorded with timestamp, actor, and detail payload
- **Model tags** — arbitrary key-value tags for filtering and organisation
- **Version comparison** — compares eval scores, metrics, aliases, and tags across versions
- **Deployment tracking** — optional deploy status with timestamps and error messages
- **Quality queries** — the latest eval score for any schema

## Quick Start

With Docker Compose (service + PostgreSQL 18):

```bash
docker compose up --build
curl -s http://localhost:8155/health/ready
```

With `uv` and a PostgreSQL you already have:

```bash
uv sync
MODEL_REGISTRY_ENVIRONMENT=development \
MODEL_REGISTRY_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5454/model_registry \
  uv run uvicorn src.main:app --host 0.0.0.0 --port 8155
```

Migrations run automatically to `head` on startup.

## API Endpoints

### Models `/v1/models`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Register a new model version |
| GET | `/` | List model versions with schema/status filters |
| GET | `/{schema_id}/quality` | Get latest quality score for a schema |
| GET | `/{schema_id}/compare` | Compare model versions for a schema |
| GET | `/{model_id}` | Retrieve model version details by UUID |
| PATCH | `/{model_id}/status` | Update model lifecycle status |
| PATCH | `/{model_id}/deploy-status` | Update deployment status |
| PUT | `/{model_id}/tags` | Set tags on a model version |
| GET | `/{model_id}/tags` | Get all tags for a model version |
| DELETE | `/{model_id}/tags/{key}` | Delete a tag from a model version |

### Tags `/v1/tags`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/search` | Search models by tag key and optional value |

### Deploy `/v1/deploy`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/{model_id}` | Trigger deployment for an approved model |

### Routing `/v1/routing`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Get the full routing table (TTL-cached) |
| GET | `/{schema_id}` | Get the routing rule for a schema |
| PUT | `/{schema_id}` | Create or update a routing rule |

### Aliases `/v1/aliases`

| Method | Path | Description |
|--------|------|-------------|
| PUT | `/{schema_id}/{alias_name}` | Set or update a model alias |
| GET | `/{schema_id}` | List all aliases for a schema |
| GET | `/{schema_id}/{alias_name}` | Get alias details |
| GET | `/{schema_id}/{alias_name}/resolve` | Resolve an alias to a full model version |
| DELETE | `/{schema_id}/{alias_name}` | Delete an alias |

### Audit `/v1/audit`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List audit events with entity/type filters |
| GET | `/models/{model_id}` | Audit history for a model |
| GET | `/schemas/{schema_id}` | Audit history for a schema |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Basic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks the database) |

## Configuration

All variables are prefixed with `MODEL_REGISTRY_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REGISTRY_SERVICE_NAME` | `model-registry` | Service name for telemetry |
| `MODEL_REGISTRY_PORT` | `8155` | HTTP listen port |
| `MODEL_REGISTRY_ENVIRONMENT` | `production` | `production` enforces the required-field check |
| `MODEL_REGISTRY_LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_REGISTRY_LOG_FORMAT` | `json` | Log format (`json` or `console`) |
| `MODEL_REGISTRY_DATABASE_URL` | `""` | PostgreSQL DSN (`postgresql+psycopg://…`); required in production |
| `MODEL_REGISTRY_ROUTING_CACHE_TTL` | `60` | Routing-table cache TTL in seconds |
| `MODEL_REGISTRY_AUTO_APPROVE_SCORE` | `0.95` | Minimum eval score for auto-approval |
| `MODEL_REGISTRY_MINIO_ENDPOINT` | `""` | S3/MinIO endpoint for GGUF storage |
| `MODEL_REGISTRY_MINIO_ACCESS_KEY` | `""` | S3/MinIO access key |
| `MODEL_REGISTRY_MINIO_SECRET_KEY` | `""` | S3/MinIO secret key |
| `MODEL_REGISTRY_GITOPS_REPO_PATH` | `<tmp>/gitops` | Local path to a GitOps clone |
| `MODEL_REGISTRY_GITOPS_ENABLED` | `false` | Enable git commit/push for GGUF deployments |
| `MODEL_REGISTRY_DEPLOYMENT_TRACKING_ENABLED` | `false` | Enable deploy-status tracking fields |
| `MODEL_REGISTRY_RTX_ENDPOINT` | `""` | vLLM endpoint for PEFT models (name kept for compatibility) |
| `MODEL_REGISTRY_ADAPTER_MANAGER_URL` | `""` | Adapter-manager sidecar URL for LoRA loading |
| `MODEL_REGISTRY_CLUSTER_DNS_DOMAIN` | `example.com` | DNS suffix used to build deployed-model endpoints |
| `MODEL_REGISTRY_DEPLOYED_SERVICE_PORT` | `8080` | Listen port of a deployed model service |
| `MODEL_REGISTRY_OTLP_TRACES_ENDPOINT` | `""` | OTLP HTTP endpoint for trace export |

## Architecture

```
src/
├── main.py                        # FastAPI app, lifespan, health endpoints
├── settings.py                    # Pydantic BaseSettings (MODEL_REGISTRY_ prefix)
├── db.py                          # async SQLAlchemy engine, session factory, migrations
├── logging_config.py              # structured logging via structlog
├── telemetry.py                   # OpenTelemetry tracing setup
├── models/
│   ├── db.py                      # SQLAlchemy ORM tables
│   └── domain.py                  # Pydantic request/response schemas
├── routers/
│   ├── models.py                  # model CRUD + tags + compare + deploy-status
│   ├── routing.py                 # routing table endpoints
│   ├── deploy.py                  # deployment trigger endpoint
│   ├── audit.py                   # audit trail query endpoints
│   └── aliases.py                 # named alias management endpoints
└── services/
    ├── model_service.py           # model version lifecycle and business logic
    ├── routing_service.py         # routing table with TTL cache
    ├── deploy_service.py          # GitOps GGUF deploy + PEFT adapter loading
    ├── audit_service.py           # audit event recording and queries
    └── alias_service.py           # champion/rollback alias management
```

## Design Principles

1. **Append-only audit** — every mutation emits an audit event in the same database transaction
2. **Cache-then-invalidate** — routing lookups use a TTL cache; writes invalidate the affected
   entry immediately
3. **Dual deployment paths** — GGUF and PEFT models share the lifecycle and routing API but
   diverge at deployment time
4. **Stateless compute** — no model weights or inference state; a metadata coordinator backed
   by PostgreSQL

## Testing

The suite requires PostgreSQL. The DSN is resolved from `TEST_DATABASE_URL` → `DATABASE_URL` →
`localhost:5432/model_registry`.

```bash
docker run -d --name mr-testpg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=model_registry -p 5454:5432 postgres:18-alpine
TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5454/model_registry" \
  uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```

## CI

`.github/workflows/ci-public.yml` runs on GitHub-hosted runners: betterleaks secret scan, the
private-reference guard, ruff, the full pytest suite against a PostgreSQL 18 service container,
and `pip-audit`. The image build/publish job is off unless the repository variable
`PUBLISH_IMAGE=true` is set.

## Docker

```bash
docker build -t model-registry .
docker run --rm -p 8155:8155 \
  -e MODEL_REGISTRY_ENVIRONMENT=development \
  -e MODEL_REGISTRY_DATABASE_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5454/model_registry \
  model-registry
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Licensing

There is no licence gate in this service: it needs no licence key, token, or network call to
build, test or run. Every feature is available by default.
