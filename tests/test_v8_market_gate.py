import numpy as np
import pandas as pd

from src.quant_trading.ml_lab.v8_market_gate import V8MarketGateResearch


def test_no_signal_days_are_zero_in_full_calendar_metrics():
    frame = pd.DataFrame(
        {
            "mean_net_return_5d": [0.10, -0.20, 0.05, -0.10],
            "hit5pct_10d": [True, False, True, False],
            "all_loss_5d": [False, True, False, True],
        }
    )

    metrics = V8MarketGateResearch.evaluate_decisions(frame, np.array([True, False, True, False]))

    assert metrics["coverage"] == 0.5
    assert metrics["all_calendar_mean_net_return"] == 0.0375
    assert metrics["eligible_mean_net_return"] == 0.075
    assert metrics["no_signal_return_policy"] == "zero_return_on_full_calendar"


def test_drawdown_improvement_uses_absolute_loss_reduction():
    assert V8MarketGateResearch.drawdown_improvement(-0.20, -0.10) == 0.5
