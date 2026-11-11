import pandas as pd

from src.quant_trading.services.data_sync import DataSyncError, DataSyncService, _asset_class_for_ts_code, _batches, _clean_float, _date_string_or_none, _raw_row_key


def test_clean_float_handles_empty_values():
    assert _clean_float(None) is None
    assert _clean_float("") is None
    assert _clean_float(float("nan")) is None


def test_clean_float_converts_numeric_values():
    assert _clean_float("1.5") == 1.5
    assert _clean_float(2) == 2.0


def test_raw_row_key_keeps_distinct_raw_financial_rows():
    left = _raw_row_key("dividend", {"ts_code": "000001.SZ", "ann_date": "20250101", "end_date": "20241231", "div_proc": "A"})
    right = _raw_row_key("dividend", {"ts_code": "000001.SZ", "ann_date": "20250101", "end_date": "20241231", "div_proc": "B"})
    assert left != right


def test_date_string_normalizes_legacy_hyphen_dates():
    assert _date_string_or_none("2025-04-29") == "20250429"
    assert _date_string_or_none("20250429.0") == "20250429"


def test_quarter_periods_cover_all_report_periods():
    periods = DataSyncService._quarter_periods("20240401", "20260601")
    assert periods[0] == "20240630"
    assert "20251231" in periods
    assert periods[-1] == "20260331"


def test_batches_preserve_order_and_bound_batch_size():
    records = [{"value": index} for index in range(2_001)]

    batches = list(_batches(records, batch_size=1_000))

    assert [len(batch) for batch in batches] == [1_000, 1_000, 1]
    assert [row["value"] for batch in batches for row in batch] == list(range(2_001))


def test_coverage_ratio_is_bounded_when_auxiliary_endpoint_has_extra_assets():
    assert DataSyncService._coverage_ratio(7_696, 5_524) == 1.0


def test_asset_class_separates_a_shares_from_funds_and_indexes():
    assert _asset_class_for_ts_code("000001.SZ") == "a_share_like"
    assert _asset_class_for_ts_code("302132.SZ") == "a_share_like"
    assert _asset_class_for_ts_code("600000.SH") == "a_share_like"
    assert _asset_class_for_ts_code("159001.SZ") == "fund_or_index_like"
    assert _asset_class_for_ts_code("000300.SH") == "fund_or_index_like"
    assert _asset_class_for_ts_code("200011.SZ") == "non_a_share_equity"
    assert _asset_class_for_ts_code("551000.SH") == "fund_or_index_like"
    assert _asset_class_for_ts_code("900901.BJ") == "fund_or_index_like"
    assert _asset_class_for_ts_code("920118.BJ") == "a_share_like"
    assert _asset_class_for_ts_code("BAD") == "unresolved"


class _V9EventClient:
    def __init__(self):
        self.range_offsets = []

    def get_top_list(self, trade_date):
        return pd.DataFrame([{"trade_date": trade_date, "ts_code": "000001.SZ"}])

    def get_top_inst(self, trade_date):
        return pd.DataFrame()

    def get_limit_list(self, trade_date):
        return pd.DataFrame([{"trade_date": trade_date, "ts_code": "000002.SZ"}])

    def get_suspend_daily(self, trade_date):
        return pd.DataFrame()

    def get_v9_event_range(self, endpoint, start_date, end_date, *, limit, offset):
        self.range_offsets.append((endpoint, offset))
        if offset:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": start_date},
                {"ts_code": "000002.SZ", "ann_date": end_date},
            ]
        )


def test_v9_event_sync_paginates_and_keeps_empty_sparse_queries_real(monkeypatch):
    client = _V9EventClient()
    service = DataSyncService(client)
    persisted = []
    runs = []
    monkeypatch.setattr(service, "_trade_dates", lambda start, end: [start, end])
    monkeypatch.setattr(service, "_upsert_raw_records", lambda endpoint, frame: persisted.append((endpoint, len(frame))) or len(frame))
    monkeypatch.setattr(service, "_record_run", lambda *args, **kwargs: runs.append((args, kwargs)))

    result = service.sync_v9_event_foundation("20260710", "20260713", page_size=2)

    assert result["failed_count"] == 0
    assert result["endpoints"]["top_inst"]["affected_rows"] == 0
    assert result["endpoints"]["share_float"]["pagination_exhausted"] is True
    assert ("share_float", 2) in client.range_offsets
    assert any(endpoint == "limit_list_d" for endpoint, _ in persisted)
    assert runs[-1][0][0] == "v9_event_foundation"


def test_v9_event_sync_rejects_unknown_endpoints():
    service = DataSyncService(_V9EventClient())

    try:
        service.sync_v9_event_foundation("20260710", "20260713", endpoints=["fake_event"])
    except DataSyncError as exc:
        assert "fake_event" in str(exc)
    else:
        raise AssertionError("unknown v9 event endpoint was accepted")


def test_v9_event_range_chunks_are_non_overlapping_and_cover_the_range():
    chunks = DataSyncService._date_range_chunks("20240101", "20240305", 30)

    assert chunks == [
        ("20240101", "20240130"),
        ("20240131", "20240229"),
        ("20240301", "20240305"),
    ]
