"""Add system_prompt to routing_rules.

Stores the system prompt injected at inference time so that training data
can include the exact same system message the model will see in production.

Revision ID: 002
Revises: 001
Create Date: 2026-03-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("routing_rules", sa.Column("system_prompt", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("routing_rules", "system_prompt")
