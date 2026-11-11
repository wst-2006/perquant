import pytest
import pandas as pd

from src.quant_trading.ml_lab.v9_events import V9EventStore


def _store():
    return V9EventStore(
        [
            {
                "endpoint": "top_list",
                "ts_code": "000001.SZ",
                "trade_date": "20250105",
                "payload": {"trade_date": "20250105", "ts_code": "000001.SZ", "net_amount": 100.0, "net_rate": 5.0},
            },
            {
                "endpoint": "top_inst",
                "ts_code": "000001.SZ",
                "trade_date": "20250105",
                "payload": {"trade_date": "20250105", "ts_code": "000001.SZ", "buy": 80.0, "sell": 20.0, "net_buy": 60.0},
            },
            {
                "endpoint": "limit_list_d",
                "ts_code": "000001.SZ",
                "trade_date": "20250105",
                "payload": {"trade_date": "20250105", "ts_code": "000001.SZ", "limit": "U", "open_times": 2, "fd_amount": 500.0},
            },
            {
                "endpoint": "repurchase",
                "ts_code": "000001.SZ",
                "ann_date": "20250105",
                "payload": {"ann_date": "20250105", "ts_code": "000001.SZ", "amount": 300.0},
            },
            {
                "endpoint": "stk_holdertrade",
                "ts_code": "000001.SZ",
                "ann_date": "20250105",
                "payload": {"ann_date": "20250105", "ts_code": "000001.SZ", "in_de": "IN", "change_ratio": 0.2},
            },
            {
                "endpoint": "share_float",
                "ts_code": "000001.SZ",
                "ann_date": "20250105",
                "payload": {"ann_date": "20250105", "ts_code": "000001.SZ", "float_date": "20250120", "float_ratio": 3.5},
            },
            {
                "endpoint": "top_list",
                "ts_code": "110047.SH",
                "trade_date": "20250105",
                "payload": {"trade_date": "20250105", "ts_code": "110047.SH", "net_amount": 999.0},
            },
        ]
    )


def test_v9_events_are_delayed_until_next_signal_date_and_filter_non_a_shares():
    rows = pd.DataFrame(
        [
            {"trade_date": "20250105", "ts_code": "000001.SZ"},
            {"trade_date": "20250106", "ts_code": "000001.SZ"},
            {"trade_date": "20250106", "ts_code": "110047.SH"},
        ]
    )

    store = _store()
    result = store.features_for_rows(rows)

    assert result.iloc[0]["feature_v9_top_list_count_20d"] == 0.0
    assert result.iloc[1]["feature_v9_top_list_count_20d"] == 1.0
    assert result.iloc[1]["feature_v9_top_list_net_amount_20d"] == 100.0
    assert result.iloc[1]["feature_v9_top_inst_buy_sell_imbalance_latest"] == pytest.approx(0.6)
    assert result.iloc[1]["feature_v9_limit_up_count_20d"] == 1.0
    assert result.iloc[1]["feature_v9_repurchase_amount_latest"] == 300.0
    assert result.iloc[1]["feature_v9_holdertrade_net_change_ratio_120d"] == pytest.approx(0.2)
    assert result.iloc[1]["feature_v9_share_float_days_until_next"] == 14.0
    assert result.iloc[1]["feature_v9_share_float_ratio_next_30d"] == 3.5
    assert result.iloc[2]["feature_v9_top_list_count_20d"] == 0.0
    assert store.filtered_non_a_share_count == 1


def test_v9_share_float_requires_known_announcement_before_signal_date():
    rows = pd.DataFrame(
        [
            {"trade_date": "20250105", "ts_code": "000001.SZ"},
            {"trade_date": "20250121", "ts_code": "000001.SZ"},
        ]
    )

    result = _store().features_for_rows(rows)

    assert result.iloc[0]["feature_v9_share_float_ratio_next_30d"] == 0.0
    assert result.iloc[1]["feature_v9_share_float_ratio_next_30d"] == 0.0
    assert pd.isna(result.iloc[1]["feature_v9_share_float_days_until_next"])
