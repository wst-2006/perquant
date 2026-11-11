import hashlib
import json

import pytest

from src.quant_trading.ml_lab.v8_daily import V8DailyPipeline, V8DailyPipelineError


class _SyncService:
    def __init__(self, *, snapshot_status="pass"):
        self.snapshot_status = snapshot_status
        self.calls = []

    def sync_stock_universe(self):
        self.calls.append("stock_universe")
        return {"task": "stock_universe", "affected_rows": 5_500}

    def sync_calendar_batch(self, start_date, end_date, datasets):
        self.calls.append(("calendar_batch", start_date, end_date, tuple(datasets)))
        return {"task": "calendar_batch", "failed_count": 0, "affected_rows": {name: 5_000 for name in datasets}}

    def stock_universe_audit(self):
        self.calls.append("stock_universe_audit")
        return {"task": "stock_universe_audit", "status": "pass", "unresolved_unknown_placeholder_count": 0}

    def trade_date_snapshot_status(self, trade_date):
        self.calls.append(("snapshot", trade_date))
        return {
            "task": "trade_date_snapshot_status",
            "status": self.snapshot_status,
            "blocking": [] if self.snapshot_status == "pass" else ["moneyflow_coverage_below_90%"],
        }

    def foundation_status(self, start_date, end_date):
        self.calls.append(("foundation", start_date, end_date))
        return {"task": "foundation_status", "status": "pass", "blocking": []}


class _InferenceService:
    def __init__(self):
        self.calls = []

    def settle(self, trade_date):
        self.calls.append(("settle", trade_date))
        return {"task": "v8_settle_paper", "outcomes_upserted": 0}

    def score_date(self, trade_date, *, persist):
        self.calls.append(("score", trade_date, persist))
        return {
            "task": "v8_score_date",
            "model_version": "test_v8",
            "decision": "no_signal",
            "paper_eligible": False,
            "diagnostic_only": True,
            "full_market_rows": 5_500,
            "diagnostic_candidates": [{"rank": rank} for rank in range(1, 6)],
            "candidates": [],
            "paper_ledger": {
                "contract": "ml_v12_paper_ledger_v1",
                "research_top5": [{"rank": rank} for rank in range(1, 6)],
                "default_70_30_portfolio": [],
                "manual_selection": {"status": "awaiting_user_review"},
                "diagnostic_only": True,
            },
            "market_gate": {"decision": "no_signal"},
            "research_gate": {"status": "observe_only"},
            "data_drift": {"status": "pass"},
            "paper_eligibility_checks": {
                "research_gate_candidate": False,
                "market_gate_eligible": False,
                "data_drift_passed": True,
            },
        }

    def status(self):
        self.calls.append("status")
        return {"task": "v8_status", "observed_signal_days": 1, "paper_signal_days": 0}


class _FactorService:
    def __init__(self, *, rows=5_400):
        self.rows = rows
        self.calls = []

    def build_snapshot(self, trade_date):
        self.calls.append(trade_date)
        return {"task": "factor_snapshot", "trade_date": trade_date, "factor_set": "factor_lab_v2", "rows": self.rows}


def test_daily_pipeline_runs_full_audited_chain_and_writes_hashed_report(tmp_path):
    sync = _SyncService()
    factor = _FactorService()
    inference = _InferenceService()
    result = V8DailyPipeline(
        report_dir=tmp_path,
        sync_service=sync,
        factor_service=factor,
        inference_service=inference,
    ).run("20260713")

    assert result["status"] == "pass"
    assert result["result"]["diagnostic_candidate_count"] == 5
    assert result["result"]["paper_candidate_count"] == 0
    assert result["result"]["paper_ledger"]["contract"] == "ml_v12_paper_ledger_v1"
    assert result["result"]["paper_ledger"]["default_70_30_portfolio"] == []
    assert result["result"]["promotion_allowed"] is False
    assert result["result"]["data_drift_status"] == "pass"
    assert inference.calls == [("settle", "20260713"), ("score", "20260713", True), "status"]
    assert factor.calls == ["20260713"]
    assert sync.calls[0] == "stock_universe"

    report_path = tmp_path / "20260713" / f"v8_daily_{result['run_id']}.json"
    payload = report_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == result["report_artifact"]["report_sha256"]
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["status"] == "pass"


def test_daily_pipeline_blocks_before_inference_when_signal_date_snapshot_is_incomplete(tmp_path):
    sync = _SyncService(snapshot_status="blocked")
    inference = _InferenceService()
    pipeline = V8DailyPipeline(report_dir=tmp_path, sync_service=sync, factor_service=_FactorService(), inference_service=inference)

    with pytest.raises(V8DailyPipelineError) as exc_info:
        pipeline.run("20260713", sync=False)

    assert exc_info.value.report["status"] == "blocked"
    assert exc_info.value.report["stages"]["trade_date_snapshot_gate"]["status"] == "blocked"
    assert inference.calls == []
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["status"] == "blocked"


def test_daily_pipeline_rejects_non_compact_dates(tmp_path):
    pipeline = V8DailyPipeline(
        report_dir=tmp_path,
        sync_service=_SyncService(),
        factor_service=_FactorService(),
        inference_service=_InferenceService(),
    )

    with pytest.raises(ValueError, match="YYYYMMDD"):
        pipeline.run("2026-07-13")


def test_daily_pipeline_blocks_when_factor_snapshot_is_not_full_market(tmp_path):
    inference = _InferenceService()
    pipeline = V8DailyPipeline(
        report_dir=tmp_path,
        sync_service=_SyncService(),
        factor_service=_FactorService(rows=999),
        inference_service=inference,
    )

    with pytest.raises(V8DailyPipelineError) as exc_info:
        pipeline.run("20260713", sync=False)

    assert exc_info.value.report["stages"]["factor_snapshot"]["status"] == "blocked"
    assert inference.calls == []
