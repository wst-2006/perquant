"""Trade-date and announcement-safe Tushare event features for v9 research."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, or_

from ..db import DailyQuote, TushareRawRecord, session_scope
from ..services.data_sync import _asset_class_for_ts_code
from .v8_events import _date, _day_number, _file_sha256, _number


V9_EVENT_CONTRACT = "ml_v9_tushare_structure_events_v1"
V9_EVENT_ENDPOINTS = (
    "top_list",
    "top_inst",
    "limit_list_d",
    "suspend_d",
    "repurchase",
    "stk_holdertrade",
    "share_float",
)
V9_TRADE_DATE_ENDPOINTS = {"top_list", "top_inst", "limit_list_d", "suspend_d"}
V9_ANNOUNCEMENT_ENDPOINTS = {"repurchase", "stk_holdertrade", "share_float"}
V9_EVENT_FEATURES = (
    "feature_v9_top_list_age_days",
    "feature_v9_top_list_count_20d",
    "feature_v9_top_list_net_amount_20d",
    "feature_v9_top_list_net_rate_latest",
    "feature_v9_top_list_net_amount_rank",
    "feature_v9_top_inst_count_20d",
    "feature_v9_top_inst_net_buy_20d",
    "feature_v9_top_inst_buy_sell_imbalance_latest",
    "feature_v9_top_inst_net_buy_rank",
    "feature_v9_limit_up_count_20d",
    "feature_v9_limit_down_count_20d",
    "feature_v9_limit_open_times_latest",
    "feature_v9_limit_fd_amount_latest",
    "feature_v9_limit_pressure_rank",
    "feature_v9_suspend_count_20d",
    "feature_v9_repurchase_age_days",
    "feature_v9_repurchase_count_120d",
    "feature_v9_repurchase_amount_latest",
    "feature_v9_repurchase_amount_rank",
    "feature_v9_holdertrade_net_change_ratio_120d",
    "feature_v9_holdertrade_increase_count_120d",
    "feature_v9_holdertrade_decrease_count_120d",
    "feature_v9_holdertrade_net_change_rank",
    "feature_v9_share_float_days_until_next",
    "feature_v9_share_float_ratio_next_30d",
    "feature_v9_share_float_ratio_next_90d",
    "feature_v9_share_float_ratio_rank",
)

_ZERO_DEFAULT_FEATURES = {
    "feature_v9_top_list_count_20d",
    "feature_v9_top_list_net_amount_20d",
    "feature_v9_top_inst_count_20d",
    "feature_v9_top_inst_net_buy_20d",
    "feature_v9_limit_up_count_20d",
    "feature_v9_limit_down_count_20d",
    "feature_v9_suspend_count_20d",
    "feature_v9_repurchase_count_120d",
    "feature_v9_holdertrade_net_change_ratio_120d",
    "feature_v9_holdertrade_increase_count_120d",
    "feature_v9_holdertrade_decrease_count_120d",
    "feature_v9_share_float_ratio_next_30d",
    "feature_v9_share_float_ratio_next_90d",
}

_RANK_FEATURES = (
    ("feature_v9_top_list_net_amount_rank", "feature_v9_top_list_net_amount_20d"),
    ("feature_v9_top_inst_net_buy_rank", "feature_v9_top_inst_net_buy_20d"),
    ("feature_v9_limit_pressure_rank", "feature_v9_limit_fd_amount_latest"),
    ("feature_v9_repurchase_amount_rank", "feature_v9_repurchase_amount_latest"),
    ("feature_v9_holdertrade_net_change_rank", "feature_v9_holdertrade_net_change_ratio_120d"),
    ("feature_v9_share_float_ratio_rank", "feature_v9_share_float_ratio_next_30d"),
)


class V9EventError(RuntimeError):
    """Raised when v9 event features cannot be built safely."""


class V9EventStore:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        deduplicated: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        filtered_non_a_share = 0
        for record in records:
            endpoint = str(record.get("endpoint") or "")
            code = str(record.get("ts_code") or "")
            if endpoint not in V9_EVENT_ENDPOINTS or not code:
                continue
            if _asset_class_for_ts_code(code) != "a_share_like":
                filtered_non_a_share += 1
                continue
            payload = record.get("payload") or {}
            public_date = _public_date(endpoint, record, payload)
            event_date = _event_date(endpoint, record, payload) or public_date
            if not public_date:
                continue
            digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
            key = endpoint, code, public_date, event_date or "", digest
            deduplicated[key] = {
                "endpoint": endpoint,
                "ts_code": code,
                "public_date": public_date,
                "public_day": _day_number(public_date),
                "event_date": event_date,
                "event_day": _day_number(event_date) if event_date else _day_number(public_date),
                "payload": payload,
            }
        self.filtered_non_a_share_count = filtered_non_a_share
        self.endpoint_counts = Counter(row["endpoint"] for row in deduplicated.values())
        self.bundles: dict[str, dict[str, dict[str, Any]]] = {endpoint: {} for endpoint in V9_EVENT_ENDPOINTS}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in deduplicated.values():
            grouped.setdefault((row["endpoint"], row["ts_code"]), []).append(row)
        for (endpoint, code), rows in grouped.items():
            ordered = sorted(rows, key=lambda item: (item["public_day"], item["event_day"], item["public_date"]))
            self.bundles[endpoint][code] = {
                "days": np.asarray([item["public_day"] for item in ordered], dtype=np.int32),
                "rows": ordered,
            }
        source_digest = sha256()
        for key, row in sorted(deduplicated.items()):
            source_digest.update(("|".join(key) + "|" + json.dumps(row["payload"], sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        self.source_sha256 = source_digest.hexdigest()

    @classmethod
    def from_database(cls, *, max_public_date: str | None = None) -> "V9EventStore":
        with session_scope() as session:
            query = session.query(
                TushareRawRecord.endpoint,
                TushareRawRecord.ts_code,
                TushareRawRecord.trade_date,
                TushareRawRecord.ann_date,
                TushareRawRecord.f_ann_date,
                TushareRawRecord.end_date,
                TushareRawRecord.payload,
            ).filter(TushareRawRecord.endpoint.in_(V9_EVENT_ENDPOINTS))
            if max_public_date:
                query = query.filter(
                    or_(
                        TushareRawRecord.trade_date <= max_public_date,
                        TushareRawRecord.ann_date <= max_public_date,
                        TushareRawRecord.f_ann_date <= max_public_date,
                    )
                )
            records = [
                {
                    "endpoint": row.endpoint,
                    "ts_code": row.ts_code,
                    "trade_date": row.trade_date,
                    "ann_date": row.ann_date,
                    "f_ann_date": row.f_ann_date,
                    "end_date": row.end_date,
                    "payload": row.payload,
                }
                for row in query.yield_per(50_000)
            ]
        return cls(records)

    @classmethod
    def from_database_window(cls, *, start_date: str, end_date: str) -> "V9EventStore":
        """Load only events that can affect signals inside an inclusive date window."""
        start = _date(start_date)
        end = _date(end_date)
        if not start or not end or start > end:
            raise V9EventError(f"Invalid event window: {start_date}-{end_date}")
        trade_start = (pd.Timestamp(start) - pd.Timedelta(days=20)).strftime("%Y%m%d")
        announcement_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y%m%d")
        float_end = (pd.Timestamp(end) + pd.Timedelta(days=90)).strftime("%Y%m%d")
        payload_ann = TushareRawRecord.payload["ann_date"].as_string()
        public_date = func.coalesce(TushareRawRecord.ann_date, TushareRawRecord.f_ann_date, payload_ann)
        float_date = TushareRawRecord.payload["float_date"].as_string()
        columns = (
            TushareRawRecord.endpoint,
            TushareRawRecord.ts_code,
            TushareRawRecord.trade_date,
            TushareRawRecord.ann_date,
            TushareRawRecord.f_ann_date,
            TushareRawRecord.end_date,
            TushareRawRecord.payload,
        )
        records: list[dict[str, Any]] = []
        with session_scope() as session:
            queries = (
                session.query(*columns).filter(
                    TushareRawRecord.endpoint.in_(V9_TRADE_DATE_ENDPOINTS),
                    TushareRawRecord.trade_date >= trade_start,
                    TushareRawRecord.trade_date <= end,
                ),
                session.query(*columns).filter(
                    TushareRawRecord.endpoint.in_({"repurchase", "stk_holdertrade"}),
                    public_date >= announcement_start,
                    public_date <= end,
                ),
                session.query(*columns).filter(
                    TushareRawRecord.endpoint == "share_float",
                    public_date <= end,
                    float_date >= start,
                    float_date <= float_end,
                ),
            )
            for query in queries:
                records.extend(_query_records(query))
        store = cls(records)
        store.query_contract = {
            "mode": "signal_window",
            "signal_start": start,
            "signal_end": end,
            "trade_event_start": trade_start,
            "announcement_event_start": announcement_start,
            "share_float_event_end": float_end,
            "same_day_policy": "features_for_rows excludes events whose public date equals the signal date",
        }
        return store

    def features_for_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not {"trade_date", "ts_code"}.issubset(rows.columns):
            raise V9EventError("Rows must include trade_date and ts_code for v9 event features.")
        output = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in V9_EVENT_FEATURES}
        for name in _ZERO_DEFAULT_FEATURES:
            output[name].fill(0.0)
        codes = rows["ts_code"].astype(str).to_numpy()
        dates = rows["trade_date"].map(_date).to_numpy()
        for index, (code, signal_date) in enumerate(zip(codes, dates, strict=False)):
            if not signal_date:
                continue
            signal_day = _day_number(signal_date)
            self._fill_top_list(output, index, code, signal_day)
            self._fill_top_inst(output, index, code, signal_day)
            self._fill_limit_list(output, index, code, signal_day)
            self._fill_suspend(output, index, code, signal_day)
            self._fill_repurchase(output, index, code, signal_day)
            self._fill_holdertrade(output, index, code, signal_day)
            self._fill_share_float(output, index, code, signal_day)
        result = pd.DataFrame(output, index=rows.index)
        dates_series = rows["trade_date"].astype(str)
        for target, source in _RANK_FEATURES:
            result[target] = pd.to_numeric(result[source], errors="coerce").groupby(dates_series, sort=False).rank(pct=True)
        return result[list(V9_EVENT_FEATURES)].astype(np.float32)

    def _fill_top_list(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._events_before("top_list", code, signal_day, lookback_days=20)
        if not events:
            return
        latest = events[-1]
        output["feature_v9_top_list_age_days"][index] = signal_day - latest["public_day"]
        output["feature_v9_top_list_count_20d"][index] = len(events)
        output["feature_v9_top_list_net_amount_20d"][index] = _sum(events, "net_amount")
        net_rate = _number(latest["payload"].get("net_rate"))
        if net_rate is not None:
            output["feature_v9_top_list_net_rate_latest"][index] = np.clip(net_rate, -1000, 1000)

    def _fill_top_inst(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._events_before("top_inst", code, signal_day, lookback_days=20)
        if not events:
            return
        latest = events[-1]
        output["feature_v9_top_inst_count_20d"][index] = len(events)
        output["feature_v9_top_inst_net_buy_20d"][index] = _sum(events, "net_buy")
        buy = _number(latest["payload"].get("buy"))
        sell = _number(latest["payload"].get("sell"))
        if buy is not None and sell is not None:
            denominator = abs(buy) + abs(sell)
            if denominator:
                output["feature_v9_top_inst_buy_sell_imbalance_latest"][index] = np.clip((buy - sell) / denominator, -1, 1)

    def _fill_limit_list(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._events_before("limit_list_d", code, signal_day, lookback_days=20)
        if not events:
            return
        latest = events[-1]
        up_count = 0
        down_count = 0
        for event in events:
            direction = _limit_direction(event["payload"])
            if direction > 0:
                up_count += 1
            elif direction < 0:
                down_count += 1
        output["feature_v9_limit_up_count_20d"][index] = up_count
        output["feature_v9_limit_down_count_20d"][index] = down_count
        open_times = _number(latest["payload"].get("open_times"))
        if open_times is not None:
            output["feature_v9_limit_open_times_latest"][index] = np.clip(open_times, 0, 1000)
        fd_amount = _number(latest["payload"].get("fd_amount") or latest["payload"].get("limit_amount"))
        if fd_amount is not None:
            output["feature_v9_limit_fd_amount_latest"][index] = np.clip(fd_amount, -1e12, 1e12)

    def _fill_suspend(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        output["feature_v9_suspend_count_20d"][index] = len(self._events_before("suspend_d", code, signal_day, lookback_days=20))

    def _fill_repurchase(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._events_before("repurchase", code, signal_day, lookback_days=120)
        if not events:
            return
        latest = events[-1]
        output["feature_v9_repurchase_age_days"][index] = signal_day - latest["public_day"]
        output["feature_v9_repurchase_count_120d"][index] = len(events)
        amount = _number(latest["payload"].get("amount"))
        if amount is not None:
            output["feature_v9_repurchase_amount_latest"][index] = np.clip(amount, 0, 1e12)

    def _fill_holdertrade(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._events_before("stk_holdertrade", code, signal_day, lookback_days=120)
        if not events:
            return
        net_change = 0.0
        increase_count = 0
        decrease_count = 0
        for event in events:
            direction = str(event["payload"].get("in_de") or "").upper()
            change = _number(event["payload"].get("change_ratio")) or 0.0
            if direction.startswith("IN"):
                net_change += abs(change)
                increase_count += 1
            elif direction.startswith("DE"):
                net_change -= abs(change)
                decrease_count += 1
        output["feature_v9_holdertrade_net_change_ratio_120d"][index] = np.clip(net_change, -1000, 1000)
        output["feature_v9_holdertrade_increase_count_120d"][index] = increase_count
        output["feature_v9_holdertrade_decrease_count_120d"][index] = decrease_count

    def _fill_share_float(self, output: dict[str, np.ndarray], index: int, code: str, signal_day: int) -> None:
        events = self._share_float_events(code, signal_day, horizon_days=90)
        if not events:
            return
        event_days = [event["event_day"] for event in events]
        output["feature_v9_share_float_days_until_next"][index] = min(event_days) - signal_day
        ratio_30d = 0.0
        ratio_90d = 0.0
        for event in events:
            ratio = _number(event["payload"].get("float_ratio")) or 0.0
            if event["event_day"] <= signal_day + 30:
                ratio_30d += ratio
            ratio_90d += ratio
        output["feature_v9_share_float_ratio_next_30d"][index] = np.clip(ratio_30d, 0, 1000)
        output["feature_v9_share_float_ratio_next_90d"][index] = np.clip(ratio_90d, 0, 1000)

    def _events_before(self, endpoint: str, code: str, signal_day: int, *, lookback_days: int) -> list[dict[str, Any]]:
        bundle = self.bundles[endpoint].get(code)
        if not bundle:
            return []
        days = bundle["days"]
        end = int(np.searchsorted(days, signal_day, side="left"))
        start = int(np.searchsorted(days, signal_day - lookback_days, side="left"))
        return bundle["rows"][start:end]

    def _share_float_events(self, code: str, signal_day: int, *, horizon_days: int) -> list[dict[str, Any]]:
        bundle = self.bundles["share_float"].get(code)
        if not bundle:
            return []
        days = bundle["days"]
        end = int(np.searchsorted(days, signal_day, side="left"))
        known = bundle["rows"][:end]
        return [event for event in known if signal_day <= event["event_day"] <= signal_day + horizon_days]


class V9EventDatasetBuilder:
    def __init__(self, store: V9EventStore) -> None:
        self.store = store

    def build(self, source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        source = Path(source_dir)
        target = Path(output_dir)
        source_manifest_path = source / "manifest.json"
        if not source_manifest_path.exists():
            raise V9EventError(f"Missing source manifest: {source_manifest_path}")
        if target.exists() and any(target.iterdir()):
            raise V9EventError(f"Output directory must be empty or absent: {target}")
        target.mkdir(parents=True, exist_ok=True)
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        rows_written = 0
        dates_written = []
        coverage = Counter()
        for path in sorted(source.glob("trade_date=*/part-000.parquet"), key=lambda item: item.parent.name):
            frame = pd.read_parquet(path)
            features = self.store.features_for_rows(frame[["trade_date", "ts_code"]])
            frame = pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
            partition = target / path.parent.name
            partition.mkdir(parents=True, exist_ok=True)
            output_path = partition / "part-000.parquet"
            frame.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
            rows_written += len(frame)
            dates_written.append(path.parent.name.split("=", 1)[1])
            coverage.update({name: int(features[name].notna().sum()) for name in V9_EVENT_FEATURES})
        if not dates_written:
            raise V9EventError(f"No trade_date partitions found in {source}")
        manifest = {
            **source_manifest,
            "task": "build_v9_event_augmented_dataset",
            "feature_set": "factor_lab_v9_events",
            "event_contract": V9_EVENT_CONTRACT,
            "event_endpoints": list(V9_EVENT_ENDPOINTS),
            "event_endpoint_rows": dict(self.store.endpoint_counts),
            "event_filtered_non_a_share_raw_rows": self.store.filtered_non_a_share_count,
            "event_source_sha256": self.store.source_sha256,
            "event_feature_columns": list(V9_EVENT_FEATURES),
            "event_feature_non_null_counts": dict(coverage),
            "parent_dataset_dir": str(source),
            "parent_manifest_sha256": _file_sha256(source_manifest_path),
            "output_dir": str(target),
            "rows": rows_written,
            "date_count": len(dates_written),
            "date_range": {"start": min(dates_written), "end": max(dates_written)},
            "availability_policy": (
                "Trade-date events are available only after their trade_date; announcement events require ann_date strictly earlier than the signal date; "
                "share_float uses announced future float_date schedules only after ann_date is known. Non-A-share raw rows are filtered out of feature generation."
            ),
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}


class V9DailyEventSnapshotBuilder:
    """Materialize immutable full-market v9 event features for a signal window."""

    def __init__(self, store: V9EventStore, *, factor_set: str = "factor_lab_v2") -> None:
        self.store = store
        self.factor_set = factor_set

    def build(self, start_date: str, end_date: str, output_dir: str | Path) -> dict[str, Any]:
        start = _date(start_date)
        end = _date(end_date)
        if not start or not end or start > end:
            raise V9EventError(f"Invalid snapshot window: {start_date}-{end_date}")
        target = Path(output_dir)
        if target.exists() and any(target.iterdir()):
            raise V9EventError(f"Output directory must be empty or absent: {target}")
        target.mkdir(parents=True, exist_ok=True)
        with session_scope() as session:
            rows = (
                session.query(DailyQuote.trade_date, DailyQuote.ts_code)
                .filter(DailyQuote.trade_date >= start, DailyQuote.trade_date <= end)
                .order_by(DailyQuote.trade_date.asc(), DailyQuote.ts_code.asc())
                .all()
            )
        keys = pd.DataFrame(
            [row for row in rows if _asset_class_for_ts_code(row.ts_code) == "a_share_like"],
            columns=["trade_date", "ts_code"],
        )
        if keys.empty:
            raise V9EventError(f"No full-market A-share quotes for {start}-{end}")
        coverage = Counter()
        date_rows: dict[str, int] = {}
        file_hashes: dict[str, str] = {}
        for trade_date, group in keys.groupby("trade_date", sort=True):
            date = str(trade_date)
            group = group.reset_index(drop=True)
            features = self.store.features_for_rows(group)
            frame = pd.concat([group, features.reset_index(drop=True)], axis=1)
            partition = target / f"trade_date={date}"
            partition.mkdir(parents=True, exist_ok=True)
            output_path = partition / "part-000.parquet"
            frame.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
            date_rows[date] = int(len(frame))
            file_hashes[date] = _file_sha256(output_path)
            coverage.update({name: int(features[name].notna().sum()) for name in V9_EVENT_FEATURES})
        dates = sorted(date_rows)
        manifest = {
            "task": "materialize_v9_daily_event_snapshots",
            "contract": "ml_v9_daily_event_snapshot_v1",
            "event_contract": V9_EVENT_CONTRACT,
            "factor_set": self.factor_set,
            "output_dir": str(target),
            "date_range": {"start": dates[0], "end": dates[-1]},
            "date_count": len(dates),
            "rows": int(sum(date_rows.values())),
            "rows_by_date": date_rows,
            "feature_columns": list(V9_EVENT_FEATURES),
            "feature_non_null_counts": dict(coverage),
            "event_endpoint_rows": dict(self.store.endpoint_counts),
            "event_filtered_non_a_share_raw_rows": self.store.filtered_non_a_share_count,
            "event_source_sha256": self.store.source_sha256,
            "event_query_contract": getattr(self.store, "query_contract", None),
            "partition_sha256": file_hashes,
            "universe_policy": "all A-share-like DailyQuote rows for each signal date; no preset stock list or candidate prefilter",
            "availability_policy": "trade-date and announcement events are strictly earlier than each signal date; share_float requires a known announcement and a future float_date",
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}


def _query_records(query: Any) -> list[dict[str, Any]]:
    return [
        {
            "endpoint": row.endpoint,
            "ts_code": row.ts_code,
            "trade_date": row.trade_date,
            "ann_date": row.ann_date,
            "f_ann_date": row.f_ann_date,
            "end_date": row.end_date,
            "payload": row.payload,
        }
        for row in query.yield_per(50_000)
    ]


def _public_date(endpoint: str, record: dict[str, Any], payload: dict[str, Any]) -> str | None:
    if endpoint in V9_TRADE_DATE_ENDPOINTS:
        return _date(record.get("trade_date") or payload.get("trade_date"))
    if endpoint in V9_ANNOUNCEMENT_ENDPOINTS:
        return _date(record.get("ann_date") or record.get("f_ann_date") or payload.get("ann_date") or payload.get("f_ann_date"))
    return None


def _event_date(endpoint: str, record: dict[str, Any], payload: dict[str, Any]) -> str | None:
    if endpoint in V9_TRADE_DATE_ENDPOINTS:
        return _date(record.get("trade_date") or payload.get("trade_date"))
    if endpoint == "share_float":
        return _date(payload.get("float_date"))
    return _date(record.get("end_date") or payload.get("end_date"))


def _sum(events: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    for event in events:
        value = _number(event["payload"].get(field))
        if value is not None:
            total += value
    return float(np.clip(total, -1e12, 1e12))


def _limit_direction(payload: dict[str, Any]) -> int:
    text = str(payload.get("limit") or payload.get("up_stat") or "").upper()
    if "D" in text or "跌" in text:
        return -1
    if "U" in text or "涨" in text:
        return 1
    pct = _number(payload.get("pct_chg"))
    if pct is None:
        return 0
    if pct > 0:
        return 1
    if pct < 0:
        return -1
    return 0
