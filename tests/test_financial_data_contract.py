from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from src.quant_trading.data.tushare_client import (
    FINA_INDICATOR_FACTOR_NAMES,
    TushareClient,
)
from src.quant_trading.factor_lab.service import _select_latest_financial_payloads
from src.quant_trading.services.data_sync import _financial_field_stock_counts


class _QueryRecorder:
    def __init__(self):
        self.calls = []

    def query(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return pd.DataFrame([{"ts_code": "000001.SZ"}])


def _client_with_recorder():
    client = object.__new__(TushareClient)
    client.pro = _QueryRecorder()
    client.request_delay_seconds = 0.0
    client.retry_times = 1
    return client


def test_financial_endpoint_requests_complete_factor_contract_by_default():
    client = _client_with_recorder()

    client.get_financial_endpoint("fina_indicator_vip", period="20260331")

    endpoint, kwargs = client.pro.calls[-1]
    requested = set(kwargs["fields"].split(","))
    assert endpoint == "fina_indicator_vip"
    assert {"ts_code", "ann_date", "end_date"}.issubset(requested)
    assert set(FINA_INDICATOR_FACTOR_NAMES).issubset(requested)


def test_financial_endpoint_preserves_explicit_field_override():
    client = _client_with_recorder()

    client.get_financial_endpoint("fina_indicator", ts_code="000001.SZ", fields="ts_code,roe")

    _, kwargs = client.pro.calls[-1]
    assert kwargs["fields"] == "ts_code,roe"


def test_latest_financial_payload_prefers_newest_period_then_richest_revision():
    rows = [
        SimpleNamespace(
            endpoint="fina_indicator",
            ts_code="000001.SZ",
            ann_date="20260430",
            end_date="20260331",
            updated_at=datetime(2026, 5, 1),
            payload={"roe": 10.0},
        ),
        SimpleNamespace(
            endpoint="fina_indicator_vip",
            ts_code="000001.SZ",
            ann_date="20260430",
            end_date="20260331",
            updated_at=datetime(2026, 4, 30),
            payload={"roe": 10.0, "roa": 5.0, "roic": 6.0},
        ),
        SimpleNamespace(
            endpoint="fina_indicator_vip",
            ts_code="000001.SZ",
            ann_date="20260701",
            end_date="20241231",
            updated_at=datetime(2026, 7, 1),
            payload={name: 1.0 for name in FINA_INDICATOR_FACTOR_NAMES},
        ),
    ]

    selected = _select_latest_financial_payloads(rows)

    assert selected["000001.SZ"] == {"roe": 10.0, "roa": 5.0, "roic": 6.0}


def test_financial_field_coverage_counts_distinct_stocks_with_real_values():
    rows = [
        SimpleNamespace(ts_code="000001.SZ", payload={"roe": 10.0, "roa": None}),
        SimpleNamespace(ts_code="000001.SZ", payload={"roe": 11.0, "roa": 5.0}),
        SimpleNamespace(ts_code="000002.SZ", payload={"roe": None, "roa": 3.0}),
    ]

    counts = _financial_field_stock_counts(rows)

    assert counts["roe"] == 1
    assert counts["roa"] == 2

    filtered = _financial_field_stock_counts(rows, {"000001.SZ"})
    assert filtered["roe"] == 1
    assert filtered["roa"] == 1
