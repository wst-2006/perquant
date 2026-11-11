import pandas as pd

from src.quant_trading.ml_lab.v8_metrics import diversified_top5, dual_head_top5, evaluate_top5, v8_promotion_gate


def _frame():
    rows = []
    for date_index, trade_date in enumerate(("20260101", "20260102")):
        for index in range(8):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"000{index:03d}.SZ",
                    "industry": "A" if index < 4 else chr(ord("B") + index - 4),
                    "prediction": float(10 - index),
                    "label_5d_net_return": 0.01 * (4 - index + date_index),
                    "label_10d_mfe_close": 0.02 * (8 - index),
                    "label_10d_hit_5pct": index < 3,
                    "label_20d_hit_8pct": index < 2,
                }
            )
    return pd.DataFrame(rows)


def test_diversified_top5_does_not_spend_all_slots_on_one_industry():
    selected = diversified_top5(_frame()[lambda df: df.trade_date == "20260101"])

    assert len(selected) == 5
    assert selected["industry"].nunique() == 5


def test_top5_metrics_include_opportunity_and_downside_measures():
    metrics = evaluate_top5(_frame())

    assert metrics["days"] == 2
    assert metrics["hit5pct_10d_at5"] == 1.0
    assert metrics["hit8pct_20d_at5"] == 1.0
    assert metrics["recall_at5"] > 0
    assert metrics["avg_industry_count"] == 5.0


def test_v11_dual_head_top5_keeps_stable_and_aggressive_slots():
    frame = _frame()[lambda df: df.trade_date == "20260101"].copy()
    frame["v11_stable_score"] = [10, 9, 8, 7, 1, 2, 3, 4]
    frame["v11_aggressive_score"] = [1, 2, 3, 4, 10, 9, 8, 7]

    selected = dual_head_top5(frame)

    assert len(selected) == 5
    assert set(selected["candidate_type"]) >= {"stable", "aggressive"}


def test_v11_evaluate_top5_uses_dual_selection_mode():
    frame = _frame().copy()
    frame["v11_stable_score"] = frame.groupby("trade_date").cumcount(ascending=False)
    frame["v11_aggressive_score"] = frame.groupby("trade_date").cumcount()

    metrics = evaluate_top5(frame, selection_mode="v11_dual_70_30")

    assert metrics["days"] == 2
    assert {row["candidate_type"] for row in metrics["selections"]} >= {"stable", "aggressive"}


def test_v8_gate_keeps_paper_only_even_when_candidate_checks_pass():
    candidate = {
        "mean_net_return_at5": 0.002,
        "positive_fold_rate": 0.7,
        "hit5pct_10d_at5": 0.7,
        "hit8pct_20d_at5": 0.5,
        "all_loss_rate_at5": 0.1,
        "max_drawdown": -0.05,
    }
    baselines = [{"hit5pct_10d_at5": 0.6, "hit8pct_20d_at5": 0.4, "all_loss_rate_at5": 0.2}]

    gate = v8_promotion_gate(candidate, baselines, audits_passed=True, regime_guard_passed=True)

    assert gate["status"] == "candidate"
    assert gate["promotion_allowed"] is False
