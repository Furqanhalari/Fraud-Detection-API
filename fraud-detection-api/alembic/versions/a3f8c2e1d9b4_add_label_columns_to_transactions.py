"""add_label_columns_to_transactions

Revision ID: a3f8c2e1d9b4
Revises: 801b027d587a
Create Date: 2026-05-02 18:00:00.000000

Adds three nullable columns to the transactions table:
  - label_source  (String)   — origin of the ground-truth label
  - label_notes   (String)   — optional free-text annotation
  - labeled_at    (DateTime) — timestamp set when is_fraud_actual is written
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a3f8c2e1d9b4"
down_revision: Union[str, Sequence[str], None] = "801b027d587a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite does not support adding multiple columns in one ALTER TABLE,
    # so each column gets its own statement.
    op.add_column("transactions", sa.Column("label_source", sa.String(), nullable=True))
    op.add_column("transactions", sa.Column("label_notes", sa.String(), nullable=True))
    op.add_column("transactions", sa.Column("labeled_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("transactions", "labeled_at")
    op.drop_column("transactions", "label_notes")
    op.drop_column("transactions", "label_source")
