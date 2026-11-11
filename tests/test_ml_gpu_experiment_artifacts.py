import json

import pandas as pd
import pytest

from scripts.build_ml_experiment_manifest import build_manifest
from scripts.verify_research_smoke import SmokeVerificationError, verify_report
from src.quant_trading.ml_lab.research import MlResearchHarness


def test_build_manifest_records_reproducibility_fields(tmp_path):
    frame_path = tmp_path / "training_frame.csv"
    pd.DataFrame(
        [
            {"trade_date": "20250102", "ts_code": "000001.SZ", "label_net_return": 0.01, "feature_a": 1.0},
            {"trade_date": "20250103", "ts_code": "000002.SZ", "label_net_return": -0.01, "feature_a": 2.0},
        ]
    ).to_csv(frame_path, index=False)

    manifest = build_manifest(frame_path, {"train_days": 8, "random_seed": 42})

    assert manifest["created_at"]
    assert manifest["training_frame_sha256"]
    assert manifest["training_frame_size_bytes"] > 0
    assert manifest["training_frame_rows"] == 2
    assert manifest["training_frame_column_count"] == 4
    assert manifest["date_range"] == {"start": "20250102", "end": "20250103"}
    assert manifest["random_seed"] == 42
    assert "/" not in manifest["python_executable"]
    assert manifest["run_mode"] == "production"
    assert manifest["diagnostic_only"] is False
    assert manifest["promotion_allowed"] is True
    assert manifest["training_frame_asset_class_counts"] == {"a_share_like": 2}


def test_build_manifest_warns_on_sensitive_column_names(tmp_path):
    frame_path = tmp_path / "training_frame.csv"
    pd.DataFrame([{"trade_date": "20250102", "account_cash": 1.0, "feature_a": 2.0}]).to_csv(frame_path, index=False)

    manifest = build_manifest(frame_path)

    assert "account_cash" in manifest["sensitive_column_warnings"]


def test_verify_research_smoke_accepts_complete_report(tmp_path):
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    payload = {
        "fold_count": 1,
        "summary": [{"model": "lightgbm"}],
        "results": [
            {"status": "trained", "target": "stable", "metrics": {"top10": {"mean_net_return": 0.01, "win_rate": 0.5, "max_drawdown": -0.01, "return_drawdown_ratio": 1.0}}},
            {"status": "trained", "target": "aggressive", "metrics": {"top10": {"mean_net_return": 0.02, "win_rate": 0.4, "max_drawdown": -0.02, "return_drawdown_ratio": 1.0}}},
        ],
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    summary_path.write_text("model,target\nlightgbm,stable\n", encoding="utf-8")

    result = verify_report(report_path, summary_path)

    assert result["success"] is True
    assert result["targets"] == ["aggressive", "stable"]


def test_verify_research_smoke_rejects_missing_target(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"fold_count": 1, "summary": [{"x": 1}], "results": []}), encoding="utf-8")

    with pytest.raises(SmokeVerificationError):
        verify_report(report_path)


def test_research_summary_marks_verdicts_as_provisional():
    summary = [
        {
            "model": "lightgbm",
            "target": "aggressive",
            "top_n": 10,
            "folds": 2,
            "positive_fold_rate": 1.0,
            "avg_mean_net_return": 0.02,
            "avg_win_rate": 0.6,
            "avg_max_drawdown": -0.01,
            "avg_return_drawdown_ratio": 2.0,
            "verdict": "usable",
        },
        {
            "model": "lightgbm",
            "target": "aggressive",
            "top_n": 20,
            "folds": 2,
            "positive_fold_rate": 0.0,
            "avg_mean_net_return": -0.01,
            "avg_win_rate": 0.4,
            "avg_max_drawdown": -0.03,
            "avg_return_drawdown_ratio": -0.3,
            "verdict": "rejected",
        },
    ]

    MlResearchHarness._apply_topn_consistency_flags(summary)

    assert all(row["verdict_is_provisional"] for row in summary)
    assert summary[0]["risk_tags"]
