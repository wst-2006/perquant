import pandas as pd

from src.quant_trading.ml_lab.failure_window_audit import ShortExecutionFailureWindowAuditor
from src.quant_trading.ml_lab.short_execution import SHORT_EXECUTION_CONTRACT


def _write_partitions(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    selections = []
    for day in range(1, 3):
        date = f"2025010{day}"
        feature_rows = []
        label_rows = []
        for stock in range(8):
            code = f"000{stock:03d}.SZ"
            feature_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "label_entry_tradable": True,
                    "feature_return_20d": float(stock),
                    "feature_market_breadth_return_5d": 0.2 * day,
                    "feature_market_breadth_flow_5d": 0.3 * day,
                    "feature_market_drawdown_20d": -0.04 * day,
                    "feature_market_return_5d": 0.01 * day,
                    "feature_market_volatility_20d": 0.02,
                    "feature_limit_down_count_5d": stock % 2,
                    "feature_limit_pressure_5d": -0.01 * stock,
                }
            )
            label_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "short_contract": SHORT_EXECUTION_CONTRACT,
                    "short_settled": True,
                    "short_realized_net_return": (stock - 3) / 100,
                    "short_positive": stock > 3,
                    "short_take_profit_hit": stock > 5,
                    "short_stop_loss_hit": stock < 2,
                    "short_gap_through_stop": stock == 0,
                    "short_ambiguous_daily_bar": False,
                    "short_exit_reason": "take_profit" if stock > 5 else "terminal_close",
                    "short_exit_untradable": False,
                    "short_blocked_exit_days": 0,
                }
            )
            if stock < 5:
                selections.append(
                    {
                        "trade_date": date,
                        "ts_code": code,
                        "industry": "A" if stock < 3 else "B",
                        "prediction": 1.0 - stock / 10,
                        "candidate_rank": stock + 1,
                        "model_disagreement": stock / 10,
                    }
                )
        for root, rows in ((features, feature_rows), (labels, label_rows)):
            part = root / f"trade_date={date}"
            part.mkdir(parents=True)
            pd.DataFrame(rows).to_parquet(part / "part-000.parquet", index=False)
    selection_path = tmp_path / "selection.csv"
    pd.DataFrame(selections).to_csv(selection_path, index=False)
    return selection_path, labels, features


def test_failure_window_audit_is_frozen_descriptive_and_writes_reports(tmp_path):
    selection_path, labels, features = _write_partitions(tmp_path)
    result = ShortExecutionFailureWindowAuditor(raw_state_factor_set=None).run(selection_path, labels, features, tmp_path / "audit")

    assert result["signal_days"] == 2
    assert result["candidate_rows"] == 10
    assert result["conclusion"]["status"] == "observe_only"
    assert result["baselines"]["random_top5_fixed_seed"]["candidate_rows"] == 10
    assert len(result["classifications"]["candidate_rank_segment"]) == 2
    assert (tmp_path / "audit" / "failure_window_audit.json").exists()
