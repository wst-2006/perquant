import json

import pandas as pd

from src.quant_trading.ml_lab.short_execution import SHORT_EXECUTION_CONTRACT
from src.quant_trading.ml_lab.short_execution_research import (
    ShortExecutionResearchConfig,
    ShortExecutionResearchHarness,
    evaluate_short_top5,
)


def _datasets(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for day in range(1, 13):
        date = f"202501{day:02d}"
        feature_rows = []
        label_rows = []
        for stock in range(30):
            signal = ((stock * 7 + day) % 30) / 30
            code = f"000{stock:03d}.SZ"
            net = signal * 0.08 - 0.03
            feature_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "industry": f"I{stock % 10}",
                    "label_entry_tradable": True,
                    "feature_return_20d": signal,
                    "feature_main_net_to_amount_sum_5d": signal * 0.5,
                }
            )
            label_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "short_settled": True,
                    "short_exit_date": f"202501{min(day + 1, 31):02d}",
                    "short_exit_reason": "take_profit" if signal > 0.45 else "terminal_close",
                    "short_terminal_target_date": f"202501{min(day + 4, 31):02d}",
                    "short_realized_net_return": net,
                    "short_positive": net > 0,
                    "short_take_profit_hit": signal > 0.45,
                    "short_stop_loss_hit": signal < 0.2,
                    "short_gap_through_stop": signal < 0.1,
                    "short_ambiguous_daily_bar": False,
                    "short_max_adverse_path": -0.08 + signal * 0.06,
                }
            )
        for root, rows in ((features, feature_rows), (labels, label_rows)):
            partition = root / f"trade_date={date}"
            partition.mkdir()
            pd.DataFrame(rows).to_parquet(partition / "part-000.parquet", index=False)
    (labels / "manifest.json").write_text(json.dumps({"contract": SHORT_EXECUTION_CONTRACT}), encoding="utf-8")
    return features, labels


def test_short_top5_metrics_observe_realized_execution_target():
    frame = pd.DataFrame(
        [
            {
                "trade_date": "20250101",
                "ts_code": f"00000{i}.SZ",
                "industry": f"I{i}",
                "prediction": float(i),
                "model_disagreement": 0.0,
                "short_realized_net_return": i / 100,
                "short_exit_reason": "take_profit",
                "short_take_profit_hit": True,
                "short_stop_loss_hit": False,
                "short_gap_through_stop": False,
            }
            for i in range(10)
        ]
    )

    result = evaluate_short_top5(frame)

    assert result["summary"]["days"] == 1
    assert result["summary"]["mean_daily_net_return"] == 0.07
    assert len(result["selections"]) == 5


def test_short_research_cpu_smoke_keeps_final_audit_separate(tmp_path):
    features, labels = _datasets(tmp_path)
    output = tmp_path / "output"

    result = ShortExecutionResearchHarness().run(
        features,
        labels,
        output,
        ShortExecutionResearchConfig(
            train_windows=(6,),
            validation_days=2,
            step_days=2,
            models=("lightgbm",),
            device="cpu",
            n_estimators=5,
            final_audit_fraction=0.34,
            min_final_audit_folds=1,
        ),
    )

    assert result["feature_count"] == 2
    assert result["split"]["status"] == "pass"
    assert result["best_model"]["model"] == "lightgbm"
    assert result["final_audit"]["folds"] == 2
    assert result["final_audit"]["signal_days"] == 4
    assert len(result["window_final_audits"]) == 1
    assert result["window_final_audits"][0]["train_window"] == 6
    assert "cumulative_return" in result["final_audit"]
    assert result["promotion_allowed"] is False
    assert (output / "short_execution_research_report.json").exists()


def test_common_windows_share_validation_dates_across_candidates():
    dates = [f"202501{day:02d}" for day in range(1, 31)]
    config = ShortExecutionResearchConfig(train_windows=(6, 9), validation_days=2, step_days=2, max_folds=3)

    windows = ShortExecutionResearchHarness.common_windows(dates, config, config.v8_config())

    assert len(windows[6]) == len(windows[9]) == 3
    assert [item["validation_dates"] for item in windows[6]] == [item["validation_dates"] for item in windows[9]]
    assert all(len(item["train_dates"]) == 6 for item in windows[6])
    assert all(len(item["train_dates"]) == 9 for item in windows[9])
