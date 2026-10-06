"""initial schema - users, predictions, prediction_probabilities, model_performance

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.Enum("user", "admin", name="userrole"), nullable=False, server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_path", sa.String(500), nullable=False),
        sa.Column("predicted_class", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), server_default="v1"),
        sa.Column("gradcam_path", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])

    op.create_table(
        "prediction_probabilities",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prediction_id", sa.String(36), sa.ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("cataract_probability", sa.Float, nullable=False),
        sa.Column("diabetic_retinopathy_probability", sa.Float, nullable=False),
        sa.Column("glaucoma_probability", sa.Float, nullable=False),
        sa.Column("normal_probability", sa.Float, nullable=False),
    )

    op.create_table(
        "model_performance",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), server_default="v1"),
        sa.Column("accuracy", sa.Float, nullable=False),
        sa.Column("precision", sa.Float, nullable=False),
        sa.Column("recall", sa.Float, nullable=False),
        sa.Column("f1_score", sa.Float, nullable=False),
        sa.Column("specificity", sa.Float, nullable=True),
        sa.Column("roc_auc", sa.Float, nullable=True),
        sa.Column("inference_time_ms", sa.Float, nullable=True),
        sa.Column("training_time_seconds", sa.Float, nullable=True),
        sa.Column("n_parameters", sa.Integer, nullable=True),
        sa.Column("model_size_kb", sa.Float, nullable=True),
        sa.Column("is_production", sa.Boolean, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("model_performance")
    op.drop_table("prediction_probabilities")
    op.drop_table("predictions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    sa.Enum(name="userrole").drop(op.get_bind(), checkfirst=True)
