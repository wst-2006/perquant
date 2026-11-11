"""Add Web jobs and immutable research runs.

Revision ID: 20260717_0001
Revises:
Create Date: 2026-07-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260717_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "web_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("dedupe_key", sa.String(256), nullable=True),
        sa.Column("trade_date", sa.String(8), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("progress_current", sa.Integer(), nullable=False),
        sa.Column("progress_total", sa.Integer(), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(36), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    for name in ("job_type", "dedupe_key", "trade_date", "status", "request_id", "created_at", "heartbeat_at"):
        op.create_index(f"ix_web_jobs_{name}", "web_jobs", [name])
    op.create_index(
        "uq_web_jobs_active_dedupe_key",
        "web_jobs",
        ["dedupe_key"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running', 'cancel_requested')"),
    )

    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("web_jobs.id"), nullable=True),
        sa.Column("trade_date", sa.String(8), nullable=False),
        sa.Column("pipeline", sa.String(64), nullable=False),
        sa.Column("execution_mode", sa.String(32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("artifact_sha256", sa.String(64), nullable=True),
        sa.Column("diagnostic_only", sa.Boolean(), nullable=False),
        sa.Column("promotion_allowed", sa.Boolean(), nullable=False),
        sa.Column("data_version", sa.JSON(), nullable=True),
        sa.Column("audit", sa.JSON(), nullable=True),
        sa.Column("market_gate", sa.JSON(), nullable=True),
        sa.Column("paper_ledger", sa.JSON(), nullable=True),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("report_artifact", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("trade_date", "pipeline", "version", name="uq_research_run_date_pipeline_version"),
    )
    for name in ("job_id", "trade_date", "pipeline", "execution_mode", "status", "decision", "model_version", "created_at"):
        op.create_index(f"ix_research_runs_{name}", "research_runs", [name])

    op.create_table(
        "research_candidates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("research_run_id", sa.String(36), sa.ForeignKey("research_runs.id"), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("ts_code", sa.String(16), nullable=False),
        sa.Column("name", sa.String(64), nullable=True),
        sa.Column("industry", sa.String(128), nullable=True),
        sa.Column("candidate_type", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(64), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("stable_score", sa.Float(), nullable=True),
        sa.Column("aggressive_score", sa.Float(), nullable=True),
        sa.Column("p_hit_5pct_10d", sa.Float(), nullable=True),
        sa.Column("expected_net_return_5d", sa.Float(), nullable=True),
        sa.Column("expected_mfe_10d", sa.Float(), nullable=True),
        sa.Column("expected_mae_10d", sa.Float(), nullable=True),
        sa.Column("model_disagreement", sa.Float(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("risk_tags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("research_run_id", "rank", name="uq_research_candidate_run_rank"),
        sa.UniqueConstraint("research_run_id", "ts_code", name="uq_research_candidate_run_stock"),
    )
    for name in ("research_run_id", "ts_code", "industry", "candidate_type", "decision"):
        op.create_index(f"ix_research_candidates_{name}", "research_candidates", [name])

    op.create_table(
        "manual_selections",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("research_candidate_id", sa.Integer(), sa.ForeignKey("research_candidates.id"), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.String(256), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("research_candidate_id", name="uq_manual_selection_candidate"),
    )
    op.create_index("ix_manual_selections_research_candidate_id", "manual_selections", ["research_candidate_id"])
    op.create_index("ix_manual_selections_selected", "manual_selections", ["selected"])

    op.create_table(
        "candidate_outcomes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("research_candidate_id", sa.Integer(), sa.ForeignKey("research_candidates.id"), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("settlement_date", sa.String(8), nullable=True),
        sa.Column("fixed_tradable", sa.Boolean(), nullable=False),
        sa.Column("net_return", sa.Float(), nullable=True),
        sa.Column("mfe_close", sa.Float(), nullable=True),
        sa.Column("mae_close", sa.Float(), nullable=True),
        sa.Column("hit_5pct", sa.Boolean(), nullable=True),
        sa.Column("hit_8pct", sa.Boolean(), nullable=True),
        sa.Column("hit_12pct", sa.Boolean(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("research_candidate_id", "horizon", name="uq_candidate_outcome_horizon"),
    )
    op.create_index("ix_candidate_outcomes_research_candidate_id", "candidate_outcomes", ["research_candidate_id"])
    op.create_index("ix_candidate_outcomes_horizon", "candidate_outcomes", ["horizon"])
    op.create_index("ix_candidate_outcomes_settlement_date", "candidate_outcomes", ["settlement_date"])


def downgrade() -> None:
    op.drop_table("candidate_outcomes")
    op.drop_table("manual_selections")
    op.drop_table("research_candidates")
    op.drop_table("research_runs")
    op.drop_table("web_jobs")
