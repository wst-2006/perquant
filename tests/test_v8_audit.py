import pandas as pd

from src.quant_trading.ml_lab.v8_audit import V8DatasetAuditor, _values_match
from src.quant_trading.ml_lab.v8_labels import V8MarketPanel


def test_reference_reconstruction_uses_next_open_and_t_plus_one_close():
    stocks = pd.DataFrame([{"ts_code": "000001.SZ", "industry": "Bank"}])
    quotes = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20250102", "open": 9.8, "close": 10.0, "high": 10.1, "low": 9.7},
            {"ts_code": "000001.SZ", "trade_date": "20250103", "open": 10.0, "close": 10.2, "high": 10.3, "low": 9.9},
            {"ts_code": "000001.SZ", "trade_date": "20250106", "open": 10.3, "close": 10.5, "high": 10.6, "low": 10.1},
        ]
    )
    limits = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20250102", "up_limit": 11.0, "down_limit": 9.0},
            {"ts_code": "000001.SZ", "trade_date": "20250103", "up_limit": 11.0, "down_limit": 9.0},
            {"ts_code": "000001.SZ", "trade_date": "20250106", "up_limit": 11.2, "down_limit": 9.2},
        ]
    )
    panel = V8MarketPanel.from_frames(quotes, limits, stocks)

    result = V8DatasetAuditor._reference_row(panel, "000001.SZ", "20250102", costs=None)

    assert result["label_entry_date"] == "20250103"
    assert result["label_1d_end_date"] == "20250106"
    assert result["label_1d_net_return"] > 0


def test_audit_value_comparison_handles_float32_and_missing_values():
    assert _values_match(0.1234568, 0.1234567, 5e-6)
    assert _values_match(pd.NA, None, 5e-6)
    assert _values_match(True, True, 5e-6)
    assert not _values_match(False, True, 5e-6)
