"""Fix nullable mismatch on model_format and adapter_loaded.

Migration 001 added these columns with server_default but without
nullable=False.  The ORM declares them as non-optional (Mapped[str] /
Mapped[bool]), so the database schema must match.

Revision ID: 003
Revises: 002
Create Date: 2026-03-24

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Back-fill any NULLs before adding the constraint.
    op.execute("UPDATE model_versions SET model_format = 'gguf' WHERE model_format IS NULL")
    op.execute("UPDATE routing_rules SET adapter_loaded = false WHERE adapter_loaded IS NULL")

    op.alter_column(
        "model_versions",
        "model_format",
        existing_type=sa.String(16),
        nullable=False,
        server_default="gguf",
    )
    op.alter_column(
        "routing_rules",
        "adapter_loaded",
        existing_type=sa.Boolean(),
        nullable=False,
        server_default="false",
    )


def downgrade() -> None:
    op.alter_column(
        "routing_rules",
        "adapter_loaded",
        existing_type=sa.Boolean(),
        nullable=True,
        server_default="false",
    )
    op.alter_column(
        "model_versions",
        "model_format",
        existing_type=sa.String(16),
        nullable=True,
        server_default="gguf",
    )
