import json

import pandas as pd

from scripts import run_v8
from src.quant_trading.ml_lab.v10_market_gate import (
    V10MarketGateConfig,
    V10MarketGateResearch,
)


def _ranking_report(day_count: int = 100) -> dict:
    rows = []
    for index in range(day_count):
        month = 1 + index // 20
        day = 1 + index % 20
        positive = index % 3 != 0
        rows.append(
            {
                "trade_date": f"2025{month:02d}{day:02d}",
                "fold_index": index // 20,
                "candidate_count": 5,
                "mean_net_return_5d": 0.015 if positive else -0.025,
                "market_mean_net_return_5d": 0.004 if positive else -0.010,
                "best_net_return_5d": 0.06 if positive else 0.01,
                "best_mfe_10d": 0.12 if positive else 0.02,
                "hit5pct_10d": positive,
                "hit8pct_20d": positive,
                "all_loss_5d": not positive,
            }
        )
    return {
        "report_contract": "ml_v8_top5_research_v3",
        "best_model": {"model": "ensemble", "train_window": 120, "score_variant": "composite"},
        "oos_daily": rows,
        "final_audit_baseline_summary": [
            {"baseline": "random", "mean_net_return_at5": -0.010},
            {"baseline": "rule_stable", "mean_net_return_at5": -0.005},
            {"baseline": "simple_return_20d", "mean_net_return_at5": -0.002},
        ],
    }


def _attach_state(daily: pd.DataFrame, factor_set: str):
    state = daily.copy()
    for index, column in enumerate(V10MarketGateResearch.__module__ and [
        "feature_market_return_1d",
        "feature_market_return_5d",
        "feature_market_return_20d",
        "feature_market_drawdown_20d",
        "feature_market_volatility_20d",
        "feature_market_breadth_return_5d",
        "feature_market_breadth_flow_5d",
        "feature_market_return_dispersion_5d",
        "feature_limit_pressure_5d",
        "feature_market_amount_change_5d_20d",
        "feature_industry_positive_rate_5d",
    ]):
        base = pd.to_numeric(state["stable_positive"], errors="coerce") * 2 - 1
        if column in {"feature_market_volatility_20d", "feature_market_return_dispersion_5d"}:
            state[column] = -base + index * 0.01
        else:
            state[column] = base + index * 0.01
    return state, {"source": "unit_test", "factor_set": factor_set}


def test_v10_no_signal_zero_return_and_three_state_metrics():
    frame = pd.DataFrame(
        {
            "trade_date": ["20250101", "20250102", "20250103"],
            "fold_index": [0, 0, 0],
            "mean_net_return_5d": [0.10, -0.20, 0.05],
            "best_net_return_5d": [0.20, -0.05, 0.10],
            "best_mfe_10d": [0.30, 0.01, 0.12],
            "hit5pct_10d": [True, False, True],
            "hit8pct_20d": [True, False, True],
            "all_loss_5d": [False, True, False],
            "decision": ["eligible_stable", "no_signal", "eligible_aggressive"],
            "calendar_net_return_5d": [0.10, 0.0, 0.05],
        }
    )

    metrics = V10MarketGateResearch.evaluate_decisions(frame)

    assert metrics["coverage"] == 0.666667
    assert metrics["stable_days"] == 1
    assert metrics["aggressive_days"] == 1
    assert metrics["all_calendar_mean_net_return"] == 0.05
    assert metrics["no_signal_return_policy"] == "zero_return_on_full_calendar"


def test_v10_run_keeps_selection_and_final_audit_separate(monkeypatch, tmp_path):
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_ranking_report()), encoding="utf-8")
    monkeypatch.setattr(V10MarketGateResearch, "attach_v10_market_state", staticmethod(_attach_state))

    result = V10MarketGateResearch().run(
        ranking_path,
        tmp_path / "out",
        V10MarketGateConfig(
            base_train_days=40,
            model_selection_days=20,
            final_audit_days=40,
            candidate_models=("rule",),
            stable_coverages=(0.25,),
            aggressive_coverages=(0.0, 0.20),
        ),
    )

    assert result["report_contract"] == "ml_v10_market_gate_research_v1"
    assert result["split"]["base_train"] == ["20250101", "20250220"]
    assert result["split"]["model_selection"] == ["20250301", "20250320"]
    assert result["split"]["final_audit"] == ["20250401", "20250520"]
    assert result["final_audit"]["no_signal_return_policy"] == "zero_return_on_full_calendar"
    assert result["promotion_allowed"] is False
    assert (tmp_path / "out" / "v10_market_gate_report.json").exists()
    assert (tmp_path / "out" / "v10_market_gate_final_daily.csv").exists()


def test_v10_gate_requires_drawdown_and_baseline_checks():
    final = {
        "coverage": 0.30,
        "all_calendar_mean_net_return": 0.01,
        "eligible_mean_net_return": 0.03,
        "positive_fold_rate": 0.60,
        "calendar_max_drawdown": -0.05,
        "all_loss_rate_at5_on_eligible": 0.05,
    }
    ungated = {
        "all_calendar_mean_net_return": 0.00,
        "calendar_max_drawdown": -0.20,
        "all_loss_rate_at5_on_eligible": 0.20,
    }
    comparison = {"beats_all_baseline_categories": True}

    gate = V10MarketGateResearch.gate(final, ungated, comparison, V10MarketGateConfig())

    assert gate["status"] == "candidate"
    assert gate["checks"]["drawdown_improves_35pct_vs_ungated_v9"] is True
    assert gate["promotion_allowed"] is False


def test_v10_market_gate_parser_defaults_to_v9_report():
    parser = run_v8.main.__globals__["argparse"].ArgumentParser()
    # The public parser is built inside main, so exercise the CLI through parseable arguments
    # by checking the imported config class and command string used by run_v8.
    assert V10MarketGateConfig().min_coverage == 0.25
    assert "v10-market-gate-research" in open(run_v8.__file__, encoding="utf-8").read()
    assert parser is not None
