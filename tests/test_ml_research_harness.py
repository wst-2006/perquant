import pandas as pd
import pytest

from src.quant_trading.ml_lab.research import MlResearchError, MlResearchHarness, WalkForwardConfig
from src.quant_trading.ml_lab.service import MlLabService


def _sample_frame(days: int = 18, stocks: int = 8) -> pd.DataFrame:
    rows = []
    for day in range(days):
        trade_date = f"2025{day // 28 + 1:02d}{day % 28 + 1:02d}"
        for stock in range(stocks):
            signal = (stock + day) % stocks
            net = (signal - 3) / 100
            buy_day = day + 2
            sell_day = day + 3
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"000{stock:03d}.SZ",
                    "label_tradable": True,
                    "label_net_return": net,
                    "label_stable_positive": int(net > 0),
                    "label_top10": int(signal >= 7),
                    "label_buy_date": f"2025{buy_day // 28 + 1:02d}{buy_day % 28 + 1:02d}",
                    "label_sell_date": f"2025{sell_day // 28 + 1:02d}{sell_day % 28 + 1:02d}",
                    "stable_score": signal * 11,
                    "aggressive_score": signal * 12,
                    "feature_return_5d": signal * 10,
                    "feature_return_20d": signal * 8,
                    "feature_volume_ratio": 1 + signal / 10,
                    "feature_main_net_to_amount": signal * 6,
                    "feature_main_net_to_circ_mv": signal * 5,
                    "feature_relative_industry_strength_20d": signal * 7,
                    "feature_distance_to_20d_high": signal * 4,
                    "feature_turnover_rate": signal * 3,
                }
            )
    return pd.DataFrame(rows)


def test_research_harness_adds_financial_cross_features():
    frame = _sample_frame()
    enriched = MlResearchHarness.add_cross_features(frame)
    assert "cross__return_5d__x__volume_ratio" in enriched.columns
    assert "cross__main_net_to_amount__x__relative_industry_strength_20d" in enriched.columns


def test_research_harness_builds_walk_forward_windows():
    frame = _sample_frame(days=18)
    config = WalkForwardConfig(train_days=8, validation_days=4, step_days=3)
    windows = MlResearchHarness.walk_forward_windows(frame["trade_date"], config)
    assert len(windows) >= 2
    assert windows[0].train_start == "20250101"
    assert windows[0].validation_start == "20250109"


def test_research_harness_runs_from_training_frame(tmp_path):
    path = tmp_path / "training_frame.csv"
    _sample_frame(days=16, stocks=10).to_csv(path, index=False, encoding="utf-8-sig")
    result = MlResearchHarness().run_from_training_frame(
        path,
        output_dir=tmp_path / "research",
        config=WalkForwardConfig(train_days=8, validation_days=4, step_days=4, top_n=(3, 5), base_models=("lightgbm",)),
    )
    assert result["fold_count"] >= 2
    assert result["feature_count"] > 8
    assert result["summary"]
    assert any(item.get("split_audit", {}).get("checked") for item in result["results"])
    assert (tmp_path / "research" / "research_harness_report.json").exists()
    assert (tmp_path / "research" / "research_harness_summary.csv").exists()


def test_training_frame_audit_checks_label_dates_and_columns():
    frame = _sample_frame(days=4, stocks=4)
    audit = MlResearchHarness.audit_training_frame(frame)
    assert audit["status"] == "pass"
    assert audit["label_date_order"]["checked"] is True
    assert audit["label_date_order"]["violations"] == 0
    assert audit["feature_count"] >= 8


def test_training_frame_audit_allows_financial_asset_growth_feature():
    frame = _sample_frame(days=4, stocks=4)
    frame["feature_assets_yoy"] = 1.0

    audit = MlResearchHarness.audit_training_frame(frame)

    assert "feature_assets_yoy" not in audit["sensitive_column_warnings"]
    assert audit["status"] == "pass"


def test_training_frame_audit_blocks_fully_missing_active_feature():
    frame = _sample_frame(days=4, stocks=4)
    frame["feature_all_missing"] = pd.NA

    audit = MlResearchHarness.audit_training_frame(frame)

    assert audit["status"] == "review"
    assert "feature_missingness_gate_blocked" in audit["issues"]
    assert audit["feature_missingness_gate"]["blocking_features"][0]["column"] == "feature_all_missing"


@pytest.mark.parametrize(
    "feature",
    [
        "feature_flow_risk_adjusted_5d",
        "feature_flow_momentum_alignment_5d_20d",
        "feature_flow_persistence_intensity",
        "feature_moneyflow_persistence_5d_10d",
        "feature_return_5d",
    ],
)
def test_export_and_research_share_optional_feature_contract(feature):
    assert MlLabService.is_optional_feature(feature) == MlResearchHarness.is_optional_feature(feature)


def test_walk_forward_split_audit_embargoes_label_overlap():
    frame = _sample_frame(days=12, stocks=4)
    config = WalkForwardConfig(train_days=8, validation_days=2, step_days=2)
    folds = MlResearchHarness.walk_forward_windows(frame["trade_date"], config)

    audit = MlResearchHarness.audit_walk_forward_splits(frame, folds)

    assert audit["status"] == "pass"
    assert audit["folds"][0]["embargoed_rows"] > 0
    assert audit["folds"][0]["output_rows"] < audit["folds"][0]["input_rows"]
    assert audit["folds"][0]["post_embargo_label_overlap_rows"] == 0


def test_research_audit_outputs_baseline_summary(tmp_path):
    path = tmp_path / "training_frame.csv"
    _sample_frame(days=16, stocks=10).to_csv(path, index=False, encoding="utf-8-sig")
    result = MlResearchHarness().run_audit_from_training_frame(
        path,
        output_dir=tmp_path / "audit",
        config=WalkForwardConfig(train_days=8, validation_days=4, step_days=4, top_n=(3, 5), base_models=()),
    )
    assert result["label_audit"]["status"] == "pass"
    assert result["baseline_summary"]
    assert result["evaluation_regimes"]["summary"]
    assert result["baseline_regime_summary"]
    assert any(row["model"] == "baseline:random" for row in result["baseline_summary"])
    assert (tmp_path / "audit" / "research_audit_report.json").exists()
    assert (tmp_path / "audit" / "baseline_summary.csv").exists()


def test_label_reconstruction_logic_matches_expected_return():
    buy = {"open": 10.0, "close": 10.5, "pct_chg": 1.0}
    sell = {"open": 11.0, "close": 11.0, "pct_chg": 2.0}
    result = MlResearchHarness._reconstruct_one_label(buy, sell, cost=0.0026)
    assert result["tradable"] is True
    assert result["label_net_return"] == pytest.approx(0.0974)


def test_label_reconstruction_blocks_limit_up_entry():
    buy = {"open": 10.0, "close": 10.5, "pct_chg": 10.0}
    sell = {"open": 11.0, "close": 11.0, "pct_chg": 2.0}
    result = MlResearchHarness._reconstruct_one_label(buy, sell, cost=0.0026)
    assert result["tradable"] is False
    assert "limit_up_entry_block" in result["reasons"]


def test_label_reconstruction_allows_matching_missing_quote_nontradable(monkeypatch):
    monkeypatch.setattr(MlResearchHarness, "_load_audit_quotes", lambda sample: {})
    frame = pd.DataFrame(
        [
            {
                "trade_date": "20250101",
                "ts_code": "000001.SZ",
                "label_buy_date": "20250102",
                "label_sell_date": "20250103",
                "label_tradable": False,
                "label_net_return": None,
            }
        ]
    )
    audit = MlResearchHarness.reconstruct_labels_from_db(frame)

    assert audit["missing_quote_rows"] == 1
    assert audit["tradable_mismatches"] == 0
    assert audit["return_mismatches"] == 0
    assert audit["status"] == "pass"


def test_sample_label_rows_requires_reconstruction_columns():
    with pytest.raises(MlResearchError):
        MlResearchHarness.sample_label_rows(pd.DataFrame([{"trade_date": "20250101"}]), sample_size=1, random_state=42)


def test_sample_label_rows_normalizes_float_like_dates_and_bool_labels():
    frame = pd.DataFrame(
        [
            {
                "trade_date": 20250101.0,
                "ts_code": "000001.SZ",
                "label_buy_date": "20250102.0",
                "label_sell_date": 20250103,
                "label_tradable": 1.0,
                "label_net_return": 0.01,
            }
        ]
    )
    sample = MlResearchHarness.sample_label_rows(frame, sample_size=1, random_state=42)
    row = sample.iloc[0]
    assert row["trade_date"] == "20250101"
    assert row["label_buy_date"] == "20250102"
    assert row["label_sell_date"] == "20250103"
    assert MlResearchHarness._coerce_label_bool(row["label_tradable"]) is True


def test_evaluation_regimes_and_regime_metrics_are_built_from_labels():
    frame = _sample_frame(days=8, stocks=6)
    frame["prediction"] = frame["stable_score"]
    regimes = MlResearchHarness.build_evaluation_regimes(frame)
    metrics = MlResearchHarness.evaluate_predictions_by_regime(frame, top_n_values=(3,), date_regimes=regimes["date_regimes"])
    assert regimes["policy"].startswith("evaluation-only")
    assert regimes["date_regimes"]
    assert "top3" in metrics
    assert sum(item["days"] for item in metrics["top3"].values()) == frame["trade_date"].nunique()


def test_failure_windows_flags_severe_fold_losses():
    result = {
        "model": "model",
        "target": "aggressive",
        "status": "trained",
        "fold": {
            "fold_index": 1,
            "validation_start": "20250101",
            "validation_end": "20250120",
        },
        "metrics": {
            "top10": {
                "mean_net_return": -0.02,
                "max_drawdown": -0.25,
                "win_rate": 0.3,
            }
        },
    }
    failures = MlResearchHarness.identify_failure_windows([result])
    assert failures
    assert failures[0]["reasons"] == ["mean_net_return_lte_-1pct", "max_drawdown_lte_-20pct"]


def test_best_summary_prefers_return_over_positive_fold_rate_for_top10():
    rows = [
        {"top_n": 10, "avg_mean_net_return": -0.001, "positive_fold_rate": 0.6, "avg_return_drawdown_ratio": 0.01},
        {"top_n": 10, "avg_mean_net_return": 0.0001, "positive_fold_rate": 0.4, "avg_return_drawdown_ratio": 0.01},
    ]

    assert MlResearchHarness._best_summary_row(rows) is rows[1]
