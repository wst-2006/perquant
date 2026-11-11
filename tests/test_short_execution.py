import pytest
import pandas as pd

from src.quant_trading.ml_lab.short_execution import (
    SHORT_EXECUTION_CONTRACT,
    ShortExecutionCostConfig,
    compute_short_execution_label,
    compute_short_execution_labels_for_panel,
)
from src.quant_trading.ml_lab.v8_labels import V8MarketPanel


NO_COST = ShortExecutionCostConfig(
    buy_commission_rate=0.0,
    sell_commission_rate=0.0,
    sell_stamp_tax_rate=0.0,
    slippage_rate=0.0,
)


def entry(**overrides):
    row = {
        "trade_date": "20260714",
        "open": 10.0,
        "close": 9.8,
        "high": 10.2,
        "low": 9.7,
        "up_limit": 11.0,
        "down_limit": 9.0,
    }
    row.update(overrides)
    return row


def day(trade_date, **overrides):
    row = {
        "trade_date": trade_date,
        "open": 10.0,
        "close": 10.0,
        "high": 10.05,
        "low": 9.8,
        "up_limit": 11.0,
        "down_limit": 9.0,
    }
    row.update(overrides)
    return row


def test_t_plus_one_lock_and_d2_take_profit():
    result = compute_short_execution_label(
        entry(high=10.5, low=9.4),
        [day("20260715", high=10.2)],
        costs=NO_COST,
    )

    assert result["contract"] == SHORT_EXECUTION_CONTRACT
    assert result["entry_date"] == "20260714"
    assert result["first_legal_sell_date"] == "20260715"
    assert result["locked_day_return"] == pytest.approx(-0.02)
    assert result["exit_date"] == "20260715"
    assert result["exit_reason"] == "take_profit"
    assert result["realized_net_return"] == pytest.approx(0.01)
    assert result["tp_before_sl"] is True


def test_same_daily_bar_uses_conservative_stop_first():
    result = compute_short_execution_label(
        entry(),
        [day("20260715", high=10.2, low=9.4)],
        costs=NO_COST,
    )

    assert result["ambiguous_daily_bar"] is True
    assert result["exit_reason"] == "same_bar_stop_first"
    assert result["realized_net_return"] == pytest.approx(-0.05)
    assert result["stop_loss_hit"] is True
    assert result["tp_before_sl"] is False


def test_gap_through_stop_uses_attainable_open_instead_of_threshold():
    result = compute_short_execution_label(
        entry(),
        [day("20260715", open=9.2, close=9.3, high=9.4, low=9.1, down_limit=9.0)],
        costs=NO_COST,
    )

    assert result["exit_reason"] == "gap_stop"
    assert result["gap_through_stop"] is True
    assert result["exit_reference_price"] == 9.2
    assert result["realized_net_return"] == pytest.approx(-0.08)


def test_down_limit_open_defers_stop_until_next_sellable_day():
    result = compute_short_execution_label(
        entry(),
        [
            day("20260715", open=9.0, close=9.0, high=9.0, low=9.0, down_limit=9.0),
            day("20260716", open=9.3, close=9.4, high=9.5, low=9.2, down_limit=8.1),
        ],
        costs=NO_COST,
    )

    assert result["exit_date"] == "20260716"
    assert result["exit_reason"] == "gap_stop"
    assert result["realized_net_return"] == pytest.approx(-0.07)
    assert result["blocked_exit_dates"] == [
        {"trade_date": "20260715", "reason": "down_limit_open_or_untradable"}
    ]


def test_terminal_d5_close_and_pending_before_terminal():
    three_days = [day("20260715"), day("20260716"), day("20260717")]
    pending = compute_short_execution_label(entry(), three_days, costs=NO_COST)
    settled = compute_short_execution_label(
        entry(),
        [*three_days, day("20260720", close=10.05)],
        costs=NO_COST,
    )

    assert pending["status"] == "pending"
    assert pending["settled"] is False
    assert settled["exit_reason"] == "terminal_close"
    assert settled["exit_date"] == "20260720"
    assert settled["realized_net_return"] == pytest.approx(0.005)


def test_terminal_limit_down_remains_pending_then_exits_on_first_sellable_close():
    first_four = [
        day("20260715"),
        day("20260716"),
        day("20260717"),
        day("20260720", open=9.0, close=9.0, high=9.0, low=9.0, down_limit=9.0),
    ]
    pending = compute_short_execution_label(entry(), first_four, costs=NO_COST)
    settled = compute_short_execution_label(
        entry(),
        [*first_four, day("20260721", open=9.7, close=9.8, high=9.9, low=9.6, down_limit=8.1)],
        costs=NO_COST,
    )

    assert pending["status"] == "pending_blocked_exit"
    assert pending["exit_tradable"] is False
    assert settled["exit_reason"] == "terminal_close_deferred"
    assert settled["exit_date"] == "20260721"


def test_limit_up_entry_is_never_treated_as_filled():
    result = compute_short_execution_label(entry(open=11.0, up_limit=11.0), [day("20260715")])

    assert result["status"] == "entry_untradable"
    assert result["entry_reason"] == "limit_up_open"
    assert result["realized_net_return"] is None
    assert result["settled"] is True


def test_costs_and_proxy_metadata_are_explicit():
    result = compute_short_execution_label(entry(), [day("20260715", high=10.2)])

    assert result["entry_time_target"] == "09:35"
    assert result["entry_price_source"] == "daily_open_proxy"
    assert "09:35 minute price unavailable" in result["proxy_warning"]
    assert result["costs"]["minimum_commission_mode"] == "notional_unknown_not_applied"
    assert result["realized_net_return"] < 0.01


def test_vectorized_panel_labels_match_reference_policy():
    bars = [
        {"trade_date": "20260713", "open": 9.9, "close": 10.0, "high": 10.1, "low": 9.8},
        {"trade_date": "20260714", "open": 10.0, "close": 9.8, "high": 10.2, "low": 9.7},
        {"trade_date": "20260715", "open": 10.0, "close": 9.6, "high": 10.2, "low": 9.4},
        {"trade_date": "20260716", "open": 9.5, "close": 9.6, "high": 9.7, "low": 9.3},
        {"trade_date": "20260717", "open": 9.6, "close": 9.7, "high": 9.8, "low": 9.5},
        {"trade_date": "20260720", "open": 9.7, "close": 9.8, "high": 9.9, "low": 9.6},
    ]
    quotes = pd.DataFrame([{"ts_code": "000001.SZ", **bar} for bar in bars])
    limits = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": bar["trade_date"], "up_limit": 11.0, "down_limit": 9.0}
            for bar in bars
        ]
    )
    panel = V8MarketPanel.from_frames(quotes, limits, pd.DataFrame([{"ts_code": "000001.SZ", "industry": "Bank"}]))
    vector = compute_short_execution_labels_for_panel(
        panel,
        pd.DataFrame([{"trade_date": "20260713", "ts_code": "000001.SZ"}]),
        costs=NO_COST,
        max_exit_scan_days=4,
    ).iloc[0]
    reference = compute_short_execution_label(
        {"trade_date": "20260714", "up_limit": 11.0, "down_limit": 9.0, **bars[1]},
        [
            {"up_limit": 11.0, "down_limit": 9.0, **bar}
            for bar in bars[2:]
        ],
        costs=NO_COST,
    )

    assert vector["short_status"] == reference["status"]
    assert vector["short_exit_date"] == reference["exit_date"]
    assert vector["short_exit_reason"] == reference["exit_reason"]
    assert vector["short_realized_net_return"] == pytest.approx(reference["realized_net_return"])
    assert vector["short_max_adverse_path"] == pytest.approx(reference["max_adverse_path"])
