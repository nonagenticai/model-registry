"""Add global_adapter table and enabled flag on routing_rules.

Substrate for the single-LoRA adapter rollout (Phase 1 of
``model-trainer/docs/single-adapter-rollout-plan.md``):

- New ``global_adapter`` table (single-row, enforced via CHECK id=1) holds
  the system-wide default LoRA adapter that the inference-router falls
  through to when no enabled per-schema rule matches.
- ``routing_rules.enabled`` flag (default TRUE) lets per-schema rules be
  flipped off without losing their history. Existing rows preserve their
  current behaviour because the default is TRUE.

This migration is intentionally substrate-only: no behaviour change for
downstream services. Phases 2-6 plug consumers into the new surfaces.

Revision ID: 005
Revises: 004
Create Date: 2026-05-07

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. routing_rules.enabled — default TRUE so existing rows stay enabled.
    op.add_column(
        "routing_rules",
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )

    # 2. global_adapter — single-row table. ``id`` defaults to 1 with a CHECK
    #    constraint so a second insert with id=1 hits the PK uniqueness and
    #    any explicit alternative id is rejected by the CHECK.
    op.create_table(
        "global_adapter",
        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("model_version_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("minio_uri", sa.String(length=1024), nullable=False),
        sa.Column("adapter_hash", sa.String(length=128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("id = 1", name="ck_global_adapter_singleton"),
    )


def downgrade() -> None:
    op.drop_table("global_adapter")
    op.drop_column("routing_rules", "enabled")
