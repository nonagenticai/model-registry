"""Add schema_fingerprint column to routing_rules.

Supports auto-registration of routing rules from
OpenAI-compatible ``response_format`` JSON schemas. The fingerprint is a
content hash of the canonical schema; it lets inference-router look up a
rule by schema content (rather than by a caller-chosen ``schema_id``) so
the same schema always resolves to the same internal uuid7.

Revision ID: 004
Revises: 003
Create Date: 2026-04-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "routing_rules",
        sa.Column("schema_fingerprint", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_routing_rules_schema_fingerprint",
        "routing_rules",
        ["schema_fingerprint"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_routing_rules_schema_fingerprint", table_name="routing_rules")
    op.drop_column("routing_rules", "schema_fingerprint")
