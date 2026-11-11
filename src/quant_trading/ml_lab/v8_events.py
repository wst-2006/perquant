"""Announcement-date-safe Tushare event features for v8 research and inference."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..db import TushareRawRecord, session_scope


V8_EVENT_CONTRACT = "ml_v8_tushare_events_v1"
V8_EVENT_ENDPOINTS = ("forecast", "express", "dividend")
V8_EVENT_FEATURES = (
    "feature_event_forecast_age_days",
    "feature_event_forecast_count_120d",
    "feature_event_forecast_change_mid",
    "feature_event_forecast_change_width",
    "feature_event_forecast_direction",
    "feature_event_forecast_change_rank",
    "feature_event_express_age_days",
    "feature_event_express_count_120d",
    "feature_event_express_profit_growth",
    "feature_event_express_diluted_roe",
    "feature_event_express_net_margin",
    "feature_event_express_profit_growth_rank",
    "feature_event_express_roe_rank",
    "feature_event_dividend_age_days",
    "feature_event_dividend_count_120d",
    "feature_event_dividend_cash_tax",
    "feature_event_dividend_implemented",
    "feature_event_dividend_cash_rank",
)


class V8EventError(RuntimeError):
    """Raised when real event features cannot be built safely."""


class AnnouncementEventStore:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        deduplicated: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for record in records:
            endpoint = str(record.get("endpoint") or "")
            code = str(record.get("ts_code") or "")
            public_date = _date(record.get("ann_date") or record.get("f_ann_date"))
            payload = record.get("payload") or {}
            if endpoint not in V8_EVENT_ENDPOINTS or not code or not public_date:
                continue
            end_date = _date(record.get("end_date") or payload.get("end_date")) or ""
            key = endpoint, code, public_date, end_date
            previous = deduplicated.get(key)
            priority = int(str(payload.get("update_flag") or "0") == "1")
            if previous is None or priority >= previous["_priority"]:
                deduplicated[key] = {
                    "endpoint": endpoint,
                    "ts_code": code,
                    "public_date": public_date,
                    "public_day": _day_number(public_date),
                    "payload": payload,
                    "_priority": priority,
                }
        self.endpoint_counts = Counter(row["endpoint"] for row in deduplicated.values())
        self.bundles: dict[str, dict[str, dict[str, Any]]] = {endpoint: {} for endpoint in V8_EVENT_ENDPOINTS}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in deduplicated.values():
            grouped.setdefault((row["endpoint"], row["ts_code"]), []).append(row)
        for (endpoint, code), rows in grouped.items():
            ordered = sorted(rows, key=lambda item: (item["public_day"], item["public_date"]))
            self.bundles[endpoint][code] = {
                "days": np.asarray([item["public_day"] for item in ordered], dtype=np.int32),
                "rows": ordered,
            }
        digest = sha256()
        for key, row in sorted(deduplicated.items()):
            digest.update(("|".join(key) + "|" + json.dumps(row["payload"], sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        self.source_sha256 = digest.hexdigest()

    @classmethod
    def from_database(cls) -> "AnnouncementEventStore":
        with session_scope() as session:
            rows = (
                session.query(TushareRawRecord)
                .filter(TushareRawRecord.endpoint.in_(V8_EVENT_ENDPOINTS))
                .all()
            )
        return cls(
            [
                {
                    "endpoint": row.endpoint,
                    "ts_code": row.ts_code,
                    "ann_date": row.ann_date,
                    "f_ann_date": row.f_ann_date,
                    "end_date": row.end_date,
                    "payload": row.payload,
                }
                for row in rows
            ]
        )

    def features_for_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        output = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in V8_EVENT_FEATURES}
        for name in (
            "feature_event_forecast_count_120d",
            "feature_event_express_count_120d",
            "feature_event_dividend_count_120d",
        ):
            output[name].fill(0.0)
        codes = rows["ts_code"].astype(str).to_numpy()
        dates = rows["trade_date"].map(_date).to_numpy()
        for index, (code, signal_date) in enumerate(zip(codes, dates, strict=False)):
            if not signal_date:
                continue
            signal_day = _day_number(signal_date)
            forecast, forecast_count = self._latest_before("forecast", code, signal_day)
            express, express_count = self._latest_before("express", code, signal_day)
            dividend, dividend_count = self._latest_before("dividend", code, signal_day)
            output["feature_event_forecast_count_120d"][index] = forecast_count
            output["feature_event_express_count_120d"][index] = express_count
            output["feature_event_dividend_count_120d"][index] = dividend_count
            if forecast:
                payload = forecast["payload"]
                low, high = _number(payload.get("p_change_min")), _number(payload.get("p_change_max"))
                output["feature_event_forecast_age_days"][index] = signal_day - forecast["public_day"]
                if low is not None and high is not None:
                    output["feature_event_forecast_change_mid"][index] = np.clip((low + high) / 2.0, -1000, 1000)
                    output["feature_event_forecast_change_width"][index] = np.clip(abs(high - low), 0, 1000)
                output["feature_event_forecast_direction"][index] = _forecast_direction(payload.get("type"))
            if express:
                payload = express["payload"]
                income = _number(payload.get("n_income"))
                previous_income = _number(payload.get("yoy_net_profit"))
                revenue = _number(payload.get("revenue"))
                output["feature_event_express_age_days"][index] = signal_day - express["public_day"]
                if income is not None and previous_income not in (None, 0.0):
                    output["feature_event_express_profit_growth"][index] = np.clip(income / abs(previous_income) - np.sign(previous_income), -10, 10)
                roe = _number(payload.get("diluted_roe"))
                if roe is not None:
                    output["feature_event_express_diluted_roe"][index] = np.clip(roe, -200, 200)
                if income is not None and revenue not in (None, 0.0):
                    output["feature_event_express_net_margin"][index] = np.clip(income / revenue, -5, 5)
            if dividend:
                payload = dividend["payload"]
                output["feature_event_dividend_age_days"][index] = signal_day - dividend["public_day"]
                cash = _number(payload.get("cash_div_tax"))
                if cash is not None:
                    output["feature_event_dividend_cash_tax"][index] = np.clip(cash, 0, 1000)
                output["feature_event_dividend_implemented"][index] = float(str(payload.get("div_proc") or "") == "实施")
        result = pd.DataFrame(output, index=rows.index)
        dates_series = rows["trade_date"].astype(str)
        for target, source in (
            ("feature_event_forecast_change_rank", "feature_event_forecast_change_mid"),
            ("feature_event_express_profit_growth_rank", "feature_event_express_profit_growth"),
            ("feature_event_express_roe_rank", "feature_event_express_diluted_roe"),
            ("feature_event_dividend_cash_rank", "feature_event_dividend_cash_tax"),
        ):
            result[target] = pd.to_numeric(result[source], errors="coerce").groupby(dates_series, sort=False).rank(pct=True)
        return result[list(V8_EVENT_FEATURES)].astype(np.float32)

    def _latest_before(self, endpoint: str, code: str, signal_day: int) -> tuple[dict[str, Any] | None, int]:
        bundle = self.bundles[endpoint].get(code)
        if not bundle:
            return None, 0
        days = bundle["days"]
        # Strictly earlier avoids treating an after-close announcement as known at that day's close.
        index = int(np.searchsorted(days, signal_day, side="left"))
        start = int(np.searchsorted(days, signal_day - 120, side="left"))
        return (bundle["rows"][index - 1] if index else None), max(index - start, 0)


class V8EventDatasetBuilder:
    def __init__(self, store: AnnouncementEventStore) -> None:
        self.store = store

    def build(self, source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        source = Path(source_dir)
        target = Path(output_dir)
        source_manifest_path = source / "manifest.json"
        if not source_manifest_path.exists():
            raise V8EventError(f"Missing source manifest: {source_manifest_path}")
        if target.exists() and any(target.iterdir()):
            raise V8EventError(f"Output directory must be empty or absent: {target}")
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
            coverage.update({name: int(features[name].notna().sum()) for name in V8_EVENT_FEATURES})
        manifest = {
            **source_manifest,
            "task": "build_v8_event_augmented_dataset",
            "feature_set": "factor_lab_v8_events",
            "event_contract": V8_EVENT_CONTRACT,
            "event_endpoints": list(V8_EVENT_ENDPOINTS),
            "event_endpoint_rows": dict(self.store.endpoint_counts),
            "event_source_sha256": self.store.source_sha256,
            "event_feature_columns": list(V8_EVENT_FEATURES),
            "event_feature_non_null_counts": dict(coverage),
            "parent_dataset_dir": str(source),
            "parent_manifest_sha256": _file_sha256(source_manifest_path),
            "output_dir": str(target),
            "rows": rows_written,
            "date_count": len(dates_written),
            "date_range": {"start": min(dates_written), "end": max(dates_written)},
            "announcement_policy": "Only announcements strictly earlier than the signal date are available; same-day publication is delayed to the next signal date.",
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}


def _date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("-", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text[:8] if len(text) >= 8 else None


def _day_number(value: str) -> int:
    return int(pd.Timestamp(value).toordinal())


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _forecast_direction(value: Any) -> float:
    text = str(value or "")
    if text in {"预增", "扭亏", "略增", "续盈"}:
        return 1.0
    if text in {"预减", "首亏", "略减", "续亏"}:
        return -1.0
    return 0.0


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()
