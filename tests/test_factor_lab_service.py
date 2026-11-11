from types import SimpleNamespace

import pandas as pd

from src.quant_trading.factor_lab.service import FactorLabService, _json_clean


def quote(trade_date, open_price, close_price, pct_chg):
    return SimpleNamespace(
        trade_date=trade_date,
        open=open_price,
        close=close_price,
        pct_chg=pct_chg,
    )


def test_forward_return_blocks_limit_up_entry():
    service = FactorLabService()
    payload = service._forward_return_payload(
        "000001.SZ",
        "20250101",
        [quote("20250102", 10.0, 10.2, 10.0), quote("20250103", 10.1, 10.3, 1.0)],
        horizon=1,
    )
    assert payload["tradable"] is False
    assert payload["net_forward_return"] is None
    assert payload["constraints"]["entry_limit_up"] is True


def test_forward_return_applies_round_trip_cost():
    service = FactorLabService()
    payload = service._forward_return_payload(
        "000001.SZ",
        "20250101",
        [quote("20250102", 10.0, 10.2, 1.0), quote("20250103", 10.1, 11.0, 1.0)],
        horizon=2,
    )
    assert payload["tradable"] is True
    assert round(payload["forward_return"], 4) == 0.1
    assert round(payload["net_forward_return"], 4) == 0.0984


def test_relative_industry_strength_subtracts_industry_median():
    raw = {"return_20d": 0.12}
    result = FactorLabService._with_relative_industry_strength(raw, 0.05)
    assert result["relative_industry_strength_20d"] == 0.06999999999999999


def test_safe_quantile_labels_handles_small_groups():
    labels = FactorLabService._safe_quantile_labels(pd.Series([1.0]), 5)
    assert pd.isna(labels.iloc[0])


def test_json_clean_converts_pandas_missing_values():
    result = _json_clean({"a": pd.NA, "b": float("nan"), "c": [1, pd.NA]})
    assert result == {"a": None, "b": None, "c": [1, None]}


def test_walk_forward_method_exists():
    assert hasattr(FactorLabService, "walk_forward_factor")


def test_factor_frame_evaluation_method_exists():
    assert hasattr(FactorLabService, "_evaluate_factor_frame")
