"""Add adapter_path and model_format to model_versions and routing_rules.

Revision ID: 001
Revises: None
Create Date: 2026-03-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_versions", sa.Column("adapter_path", sa.String(1024), nullable=True))
    op.add_column("model_versions", sa.Column("model_format", sa.String(16), server_default="gguf"))
    op.add_column("routing_rules", sa.Column("adapter_path", sa.String(1024), nullable=True))
    op.add_column(
        "routing_rules", sa.Column("adapter_loaded", sa.Boolean(), server_default="false")
    )


def downgrade() -> None:
    op.drop_column("routing_rules", "adapter_loaded")
    op.drop_column("routing_rules", "adapter_path")
    op.drop_column("model_versions", "model_format")
    op.drop_column("model_versions", "adapter_path")
