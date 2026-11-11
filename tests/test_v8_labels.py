import pandas as pd

from src.quant_trading.ml_lab.v8_labels import V8MarketPanel, compute_path_labels


def test_path_labels_obey_t_plus_one_and_stock_specific_limits():
    labels = compute_path_labels(
        {"trade_date": "20260713", "open": 10.0, "up_limit": 11.0},
        [
            {"trade_date": "20260714", "close": 10.5, "high": 10.8, "down_limit": 9.0},
            {"trade_date": "20260715", "close": 10.8, "high": 11.0, "down_limit": 9.2},
            {"trade_date": "20260716", "close": 10.4, "high": 10.9, "down_limit": 9.5},
        ],
        horizons=(1, 3),
        thresholds=(0.05, 0.08),
    )

    assert labels["label_entry_tradable"] is True
    assert labels["label_1d_end_date"] == "20260714"
    assert labels["label_3d_mfe_close"] > labels["label_1d_mfe_close"]
    assert labels["label_3d_hit_5pct"] is True
    assert labels["label_3d_time_to_5pct"] == 2


def test_limit_up_open_blocks_all_opportunity_labels():
    labels = compute_path_labels(
        {"trade_date": "20260713", "open": 11.0, "up_limit": 11.0},
        [{"trade_date": "20260714", "close": 12.0, "high": 12.0, "down_limit": 9.0}],
        horizons=(1,),
        thresholds=(0.05,),
    )

    assert labels["label_entry_tradable"] is False
    assert labels["label_entry_reason"] == "limit_up_open"
    assert labels["label_1d_net_return"] is None
    assert labels["label_1d_hit_5pct"] is None


def test_limit_down_close_is_not_counted_as_realizable_mfe():
    labels = compute_path_labels(
        {"trade_date": "20260713", "open": 10.0, "up_limit": 11.0},
        [{"trade_date": "20260714", "close": 9.0, "high": 10.5, "down_limit": 9.0}],
        horizons=(1,),
        thresholds=(0.05,),
    )

    assert labels["label_1d_fixed_tradable"] is False
    assert labels["label_1d_mfe_close"] is None
    assert labels["label_1d_mfe_high_diagnostic"] is not None


def test_vectorized_panel_keeps_suspension_as_missing_instead_of_skipping_date():
    stocks = pd.DataFrame([{"ts_code": "000001.SZ", "industry": "Bank"}])
    quotes = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260710", "open": 10, "close": 10, "high": 10, "low": 10},
            {"ts_code": "000001.SZ", "trade_date": "20260713", "open": 10, "close": 10, "high": 10, "low": 10},
            {"ts_code": "000001.SZ", "trade_date": "20260715", "open": 10, "close": 11, "high": 11, "low": 10},
            {"ts_code": "000002.SZ", "trade_date": "20260714", "open": 8, "close": 8, "high": 8, "low": 8},
        ]
    )
    limits = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260713", "up_limit": 11, "down_limit": 9},
            {"ts_code": "000001.SZ", "trade_date": "20260715", "up_limit": 11, "down_limit": 9},
            {"ts_code": "000002.SZ", "trade_date": "20260714", "up_limit": 9, "down_limit": 7},
        ]
    )
    panel = V8MarketPanel.from_frames(quotes, limits, stocks)
    result = panel.labels_for_rows(pd.DataFrame([{"trade_date": "20260710", "ts_code": "000001.SZ"}]), horizons=(1,))

    assert result.iloc[0]["label_entry_date"] == "20260713"
    assert pd.isna(result.iloc[0]["label_1d_net_return"])
    assert result.iloc[0]["label_1d_end_date"] == "20260714"
