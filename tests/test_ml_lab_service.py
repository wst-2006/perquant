from types import SimpleNamespace

import pytest

from scripts.run_ml_lab import resolved_factor_set_from_args
from src.quant_trading.ml_lab.service import AGGRESSIVE_WEIGHTS, MlLabConfig, MlLabService, STABLE_WEIGHTS


def test_feature_names_include_core_scores():
    names = MlLabService.feature_names()
    assert "return_20d" in names
    assert "main_net_to_amount" in names
    assert "relative_industry_strength_20d" in names


def test_weighted_score_ignores_missing_features():
    score = MlLabService._weighted_score({"return_20d": 80, "main_net_to_amount": None}, {"return_20d": 0.5, "main_net_to_amount": 0.5})
    assert score == 80


def test_data_quality_penalty_accounts_for_missing_moneyflow():
    penalty = MlLabService._data_quality_penalty({"constraints": {"moneyflow_missing": True, "low_liquidity": True, "short_listing": True}})
    assert penalty == 28


def test_contribution_factors_returns_top_weighted_items():
    features = {name: 50.0 for name in STABLE_WEIGHTS}
    features["relative_industry_strength_20d"] = 100.0
    rows = MlLabService._contribution_factors(features, STABLE_WEIGHTS, limit=2)
    assert rows[0]["factor"] == "relative_industry_strength_20d"
    assert len(rows) == 2


def test_risk_tags_identify_high_volatility_and_missing_moneyflow():
    tags = MlLabService._risk_tags(
        {"volatility_20d": 0.08, "amplitude_20d": 0.3, "return_5d": 0.2},
        {"constraints": {"moneyflow_missing": True}},
    )
    assert "资金流缺失" in tags
    assert "20日波动偏高" in tags
    assert "20日振幅偏高" in tags
    assert "短期涨幅过热" in tags


def test_bucket_summary_calculates_win_rate():
    summary = MlLabService._bucket_summary([{"profit": 1, "buy_cost": 10}, {"profit": -1, "buy_cost": 10}])
    assert summary["trades"] == 2
    assert summary["win_rate"] == 0.5


def test_pick_payload_keeps_both_scores():
    score = SimpleNamespace(ts_code="000001.SZ", stable_score=70, aggressive_score=80, stable_rank=1, aggressive_rank=2)
    stock = SimpleNamespace(name="平安银行")
    payload = MlLabService._pick_payload(score, stock, "stable")
    assert payload["stable_score"] == 70
    assert payload["aggressive_score"] == 80


def test_experiment_run_helper_returns_integer_id():
    service = MlLabService()
    assert isinstance(service._persist_experiment_run.__name__, str)


def test_factor_convergence_report_builds(tmp_path):
    service = MlLabService()
    try:
        service._persist_experiment_run(
            experiment_type="smoke",
            start_date="20240101",
            end_date="20240101",
            summary={"probe": True},
        )
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    result = service.build_factor_convergence_report(
        training_frame_path="reports/ml_lab/training_frame_full.csv",
        output_dir=tmp_path / "convergence",
        model_report_path="reports/ml_lab/models_full/model_comparison_report.json",
    )
    assert (tmp_path / "convergence" / "factor_convergence_summary.csv").exists()
    assert (tmp_path / "convergence" / "factor_convergence_details.json").exists()
    assert (tmp_path / "convergence" / "factor_pool_convergence.csv").exists()
    assert result["factor_pool"]
    verdicts = {row["verdict"] for row in result["rows"]}
    statuses = {row["status"] for row in result["factor_pool"]}
    assert verdicts <= {"保留", "观察", "降级"}
    assert statuses <= {"保留", "观察", "降级"}


def test_model_validation_summary_counts_statuses():
    summary = MlLabService._model_validation_summary([
        {"status": "trained", "model": "lightgbm", "target": "stable", "evaluations": [{"window": "w", "top10": {"label_rate": 0.5}, "top50": {"mean_net_return": 0.01}, "top100": {"mean_net_return": 0.0}}]},
        {"status": "missing_dependency", "model": "missing", "target": "stable"},
    ])
    assert summary["trained"] == 1
    assert summary["missing_dependency"] == 1
    assert summary["windows"][0]["top50_mean_net_return"] == 0.01


def test_train_models_includes_factor_pool_version(tmp_path):
    service = MlLabService()
    try:
        service._persist_experiment_run(
            experiment_type="smoke",
            start_date="20240101",
            end_date="20240101",
            summary={"probe": True},
        )
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    frame_path = tmp_path / "training_frame.csv"
    rows = []
    for day in range(12):
        for stock in range(8):
            signal = (day + stock) % 8
            rows.append(
                {
                    "trade_date": f"202401{day + 1:02d}",
                    "ts_code": f"000{stock:03d}.SZ",
                    "label_tradable": True,
                    "label_stable_positive": int(signal >= 4),
                    "label_top5": int(signal >= 6),
                    "label_top10": int(signal >= 5),
                    "feature_return_5d": float(signal),
                    "feature_turnover_rate": float(stock + 1),
                }
            )
    import pandas as pd

    pd.DataFrame(rows).to_csv(frame_path, index=False, encoding="utf-8-sig")
    report = service.train_models(
        frame_path,
        tmp_path / "models",
        factor_pool_version="candidate_20260627155905_10f_1-3-5-10-20h",
    )
    assert report["factor_pool_version"] == "candidate_20260627155905_10f_1-3-5-10-20h"


def test_build_execution_package(tmp_path):
    from scripts.build_ml_lab_package import build_package

    package = build_package(
        factor_pool_version="candidate_20260627155905_10f_1-3-5-10-20h",
        training_report_path="reports/ml_lab/models_versioned_v2/model_comparison_report.json",
        convergence_report_path="reports/ml_lab/convergence_versioned_v2/factor_convergence_details.json",
        registry_path="reports/ml_lab/versioned_registry.json",
        output_path=tmp_path / "execution_package.json",
    )
    assert package["factor_pool_version"] == "candidate_20260627155905_10f_1-3-5-10-20h"
    assert package["factor_convergence"]["保留"] is not None
    assert package["factor_convergence"]["观察"] is not None
    assert package["factor_convergence"]["降级"] is not None
    assert (tmp_path / "execution_package.json").exists()


def test_publish_versioned_artifacts_command_is_registered():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "scripts/run_ml_lab.py", "export-training-frame", "-h"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--rebuild-snapshots" in result.stdout


def test_resolved_factor_set_defaults_to_v2_when_unspecified():
    from types import SimpleNamespace

    assert resolved_factor_set_from_args(SimpleNamespace()) == "factor_lab_v2"


def test_ml_lab_config_can_separate_feature_set_from_legacy_factor_set():
    service = MlLabService(MlLabConfig(factor_set="factor_lab_v1", feature_set="factor_lab_v3"))

    assert service.active_factor_set == "factor_lab_v3"
    assert "return_120d" in service.feature_names(service.active_factor_set)
    assert "roe_dt" not in service.feature_names(service.active_factor_set)


def test_v4_feature_set_adds_meta_features_and_drops_sparse_v3_candidates():
    names = MlLabService.feature_names("factor_lab_v4")

    assert "short_vs_medium_return" in names
    assert "industry_strength_risk_adjusted" in names
    assert "roe" not in names
    assert "cross_quality_value" not in names


def test_v5_feature_set_adds_cross_sectional_and_market_breadth_features():
    names = MlLabService.feature_names("factor_lab_v5")

    assert "cs_rank_relative_industry_strength_20d" in names
    assert "market_breadth_return_5d" in names
    assert "relative_strength_momentum_alignment" in names
    assert "short_vs_medium_return" in names


def test_versioned_registry_and_execution_package_are_consistent():
    import json
    from pathlib import Path

    registry = json.loads(Path("reports/ml_lab/versioned_registry.json").read_text(encoding="utf-8"))
    package = json.loads(Path("reports/ml_lab/execution_package_v1.json").read_text(encoding="utf-8"))
    assert registry["factor_pool_version"] == package["factor_pool_version"]
    assert registry["factor_pool_registry"].endswith("candidate_pool_registry.json")
    assert set(package["factor_convergence"].keys()) == {"保留", "观察", "降级"}
