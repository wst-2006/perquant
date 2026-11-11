import json

import pandas as pd

from src.quant_trading.ml_lab.market_state import MarketStateConfig, MarketStateResearch
from src.quant_trading.ml_lab.research import MARKET_STATE_FEATURES


def _oos_rows() -> list[dict]:
    rows = []
    for fold in range(4):
        for day in range(3):
            positive = (fold + day) % 2 == 0
            row = {
                "trade_date": f"20250{fold + 1}{day + 1:02d}",
                "fold_index": fold,
                "mean_net_return": 0.012 if positive else -0.008,
                "selection_positive": int(positive),
            }
            for index, feature in enumerate(MARKET_STATE_FEATURES):
                row[feature] = float(fold * 3 + day + index + 1)
            rows.append(row)
    return rows


def test_market_state_research_uses_only_prior_oos_folds_and_zeroes_no_signal_days(tmp_path):
    ranking = tmp_path / "ranking.json"
    audit = tmp_path / "audit.json"
    ranking.write_text(json.dumps({"oos_daily_selection": _oos_rows()}), encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "baseline_comparison": {
                    "best_baseline_by_category": {
                        "random": {"avg_mean_net_return": -0.001},
                        "rule": {"avg_mean_net_return": -0.002},
                        "simple_factor": {"avg_mean_net_return": -0.003},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    research = MarketStateResearch()
    research.attach_raw_market_state = lambda daily, factor_set: (daily, {"source": "test", "coverage": {feature: 1.0 for feature in MARKET_STATE_FEATURES}})
    result = research.run(
        ranking,
        audit,
        tmp_path / "output",
        MarketStateConfig(min_history_days=3, target_coverage=0.25, min_coverage=0.25),
    )

    rule_rows = pd.DataFrame(result["daily_results"]["rule"])
    assert (rule_rows[rule_rows["fold_index"] == 0]["decision"] == "no_signal").all()
    assert (rule_rows.loc[rule_rows["decision"] == "no_signal", "calendar_return"] == 0.0).all()
    assert (tmp_path / "output" / "market_state_report.json").exists()


def test_market_state_rejects_duplicate_oos_dates():
    rows = pd.DataFrame(_oos_rows())
    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    try:
        MarketStateResearch.prepare_daily_rows(duplicated)
    except Exception as exc:
        assert "duplicated trade dates" in str(exc)
    else:
        raise AssertionError("duplicate OOS dates must block market-state research")


def test_market_state_gate_compares_drawdown_to_unconditional_calendar_baseline():
    summaries = [
        {"method": "unconditional_v5", "calendar_max_drawdown": -0.40},
        {"method": "lightgbm", "coverage_rate": 0.30, "calendar_mean_net_return": 0.01, "eligible_mean_net_return": 0.02, "positive_fold_rate": 0.60, "calendar_max_drawdown": -0.20},
    ]

    gate = MarketStateResearch.gate(summaries, {"beats_all_baseline_categories": True}, MarketStateConfig())

    assert gate["reference_drawdown"] == -0.4
    assert gate["checks"]["drawdown_improved"] is True
