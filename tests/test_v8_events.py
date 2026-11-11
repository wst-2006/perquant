import pandas as pd

from src.quant_trading.ml_lab.v8_events import AnnouncementEventStore


def _store():
    return AnnouncementEventStore(
        [
            {
                "endpoint": "forecast",
                "ts_code": "000001.SZ",
                "ann_date": "20250105",
                "end_date": "20241231",
                "payload": {"type": "预增", "p_change_min": 20.0, "p_change_max": 40.0},
            },
            {
                "endpoint": "express",
                "ts_code": "000001.SZ",
                "ann_date": "20250106",
                "end_date": "20241231",
                "payload": {"n_income": 120.0, "yoy_net_profit": 100.0, "revenue": 1000.0, "diluted_roe": 8.0},
            },
        ]
    )


def test_same_day_announcement_is_not_available_until_next_signal_date():
    rows = pd.DataFrame(
        [
            {"trade_date": "20250105", "ts_code": "000001.SZ"},
            {"trade_date": "20250106", "ts_code": "000001.SZ"},
        ]
    )

    result = _store().features_for_rows(rows)

    assert pd.isna(result.iloc[0]["feature_event_forecast_change_mid"])
    assert result.iloc[1]["feature_event_forecast_change_mid"] == 30.0
    assert result.iloc[1]["feature_event_forecast_direction"] == 1.0
    assert pd.isna(result.iloc[1]["feature_event_express_profit_growth"])


def test_event_absence_remains_missing_while_counts_remain_real_zero():
    rows = pd.DataFrame([{"trade_date": "20250110", "ts_code": "000002.SZ"}])

    result = _store().features_for_rows(rows)

    assert pd.isna(result.iloc[0]["feature_event_forecast_age_days"])
    assert result.iloc[0]["feature_event_forecast_count_120d"] == 0.0
