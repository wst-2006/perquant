from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from ..data.tushare_client import (
    FINA_INDICATOR_FACTOR_NAMES,
    FINA_INDICATOR_GATE_FIELDS,
    FINA_INDICATOR_STRUCTURALLY_SPARSE_FIELDS,
    TushareClient,
)
from ..db import DailyBasic, DailyQuote, DataSyncRun, IndexDaily, MoneyflowDaily, Stock, StockAdjFactor, StockLimitDaily, TushareRawRecord, session_scope
from ..utils.dates import today_yyyymmdd


class DataSyncError(RuntimeError):
    """Raised when a real data synchronization task cannot complete."""


CORE_INDEX_CODES = ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH", "000016.SH")
INDEX_BASIC_MARKETS = ("SSE", "SZSE", "CSI", "SW", "MSCI", "CICC", "OTH")
FINANCIAL_ENDPOINTS = ("income", "balancesheet", "cashflow", "fina_indicator", "forecast", "express", "dividend", "fina_audit", "fina_mainbz")
V9_EVENT_ENDPOINTS = ("top_list", "top_inst", "limit_list_d", "suspend_d", "repurchase", "stk_holdertrade", "share_float")
V9_DAILY_EVENT_ENDPOINTS = {"top_list", "top_inst", "limit_list_d", "suspend_d"}
V9_RANGE_EVENT_ENDPOINTS = set(V9_EVENT_ENDPOINTS) - V9_DAILY_EVENT_ENDPOINTS
V9_EVENT_PAGE_SIZE = 6000
V9_EVENT_MAX_PAGES = 100
V9_EVENT_MAX_OFFSET = 100_000
V9_EVENT_RANGE_CHUNK_DAYS = 30
VIP_FINANCIAL_ENDPOINTS = {"income", "balancesheet", "cashflow", "fina_indicator"}
FINA_MAINBZ_TYPES = ("P", "D", "I")
RAW_COVERAGE_REQUIRED_ENDPOINTS = (
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
    "dividend",
    "fina_audit",
    "fina_mainbz_P",
    "fina_mainbz_D",
    "fina_mainbz_I",
)
MIN_RAW_STOCK_COVERAGE = 0.90
MIN_FINANCIAL_FIELD_STOCK_COVERAGE = 0.90
LEGACY_RAW_FINANCIAL_TABLES = {
    "income": "raw_income",
    "balancesheet": "raw_balancesheet",
    "cashflow": "raw_cashflow",
    "fina_indicator": "raw_fina_indicator",
    "dividend": "raw_dividend",
}
UPSERT_BATCH_SIZE = 1_000


def _asset_class_for_ts_code(ts_code: str | None) -> str:
    """Classify placeholders so non-stock assets do not pollute A-share universes."""
    if not ts_code or "." not in str(ts_code):
        return "unresolved"
    symbol, suffix = str(ts_code).upper().split(".", 1)
    if not symbol.isdigit():
        return "unresolved"
    if suffix == "BJ":
        if symbol.startswith(("43", "83", "87", "88", "92")):
            return "a_share_like"
        return "fund_or_index_like"
    if suffix == "SZ":
        if symbol.startswith(("000", "001", "002", "003", "300", "301", "302")):
            return "a_share_like"
        if symbol.startswith(("200", "201")):
            return "non_a_share_equity"
        if symbol.startswith(("15", "16", "17", "18", "39")):
            return "fund_or_index_like"
        return "unresolved"
    if suffix == "SH":
        if symbol.startswith(("600", "601", "603", "605", "688", "689")):
            return "a_share_like"
        if symbol.startswith("900"):
            return "non_a_share_equity"
        if symbol.startswith(("000", "50", "51", "52", "53", "55", "56", "58", "90", "93", "88")):
            return "fund_or_index_like"
        return "unresolved"
    return "unresolved"


def _clean_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _date_string_or_none(value: Any) -> str | None:
    text_value = _string_or_none(value)
    if text_value is None:
        return None
    text_value = text_value.strip()
    if text_value.endswith(".0") and text_value[:-2].isdigit():
        text_value = text_value[:-2]
    compact = text_value.replace("-", "").replace("/", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        return compact[:8]
    return text_value[:8]


def _json_payload(row: dict[str, Any]) -> dict:
    payload = {}
    for key, value in row.items():
        if value is None:
            payload[key] = None
            continue
        try:
            if pd.isna(value):
                payload[key] = None
                continue
        except TypeError:
            pass
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)):
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def _raw_row_key(endpoint: str, payload: dict) -> str:
    key_fields = [
        "ts_code",
        "trade_date",
        "ann_date",
        "f_ann_date",
        "end_date",
        "period",
        "report_type",
        "comp_type",
        "type",
        "name",
        "bz_item",
    ]
    parts = [str(payload.get(field)) for field in key_fields if payload.get(field) not in (None, "")]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    if not parts:
        parts = [digest]
    else:
        parts.append(digest)
    text = "|".join([endpoint, *parts])
    if len(text) <= 240:
        return text
    return f"{endpoint}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _financial_field_stock_counts(rows: list[Any], universe_codes: set[str] | None = None) -> dict[str, int]:
    covered = {name: set() for name in FINA_INDICATOR_FACTOR_NAMES}
    for row in rows:
        ts_code = getattr(row, "ts_code", None)
        payload = getattr(row, "payload", None)
        if not ts_code or not isinstance(payload, dict):
            continue
        code = str(ts_code)
        if universe_codes is not None and code not in universe_codes:
            continue
        for name in FINA_INDICATOR_FACTOR_NAMES:
            if _clean_float(payload.get(name)) is not None:
                covered[name].add(code)
    return {name: len(codes) for name, codes in covered.items()}


def _batches(records: list[dict[str, Any]], batch_size: int = UPSERT_BATCH_SIZE):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


@dataclass
class DataSyncService:
    client: TushareClient | None

    @classmethod
    def create(cls) -> "DataSyncService":
        return cls(TushareClient())

    def sync_stock_pool(self) -> dict:
        df = self.client.get_all_a_shares(with_filter=False)
        if df.empty:
            raise DataSyncError("Tushare returned an empty full-market stock pool.")
        affected = self._upsert_stocks(df)
        self._record_run("stock_pool", "success", affected_rows=affected)
        return {"task": "stock_pool", "affected_rows": affected}

    def sync_stock_universe(self) -> dict:
        df = self.client.get_full_stock_universe()
        if df.empty:
            raise DataSyncError("Tushare returned an empty full stock universe.")
        affected = self._upsert_stocks(df)
        self._record_run("stock_universe", "success", affected_rows=affected)
        return {"task": "stock_universe", "affected_rows": affected}

    def sync_daily_quotes(self, trade_date: str | None = None) -> dict:
        date_value = trade_date or today_yyyymmdd()
        df = self.client.get_daily_quotes(trade_date=date_value)
        if df.empty:
            raise DataSyncError(f"Tushare daily returned no rows for trade_date={date_value}.")
        affected = self._upsert_daily_quotes(df)
        self._record_run("daily_quotes", "success", trade_date=date_value, affected_rows=affected)
        return {"task": "daily_quotes", "trade_date": date_value, "affected_rows": affected}

    def sync_daily_basic(self, trade_date: str | None = None) -> dict:
        date_value = trade_date or today_yyyymmdd()
        df = self.client.get_daily_basic(trade_date=date_value)
        if df.empty:
            raise DataSyncError(f"Tushare daily_basic returned no rows for trade_date={date_value}.")
        affected = self._upsert_daily_basic(df)
        self._record_run("daily_basic", "success", trade_date=date_value, affected_rows=affected)
        return {"task": "daily_basic", "trade_date": date_value, "affected_rows": affected}

    def sync_calendar_batch(self, start_date: str, end_date: str, datasets: list[str] | None = None) -> dict:
        selected = datasets or ["daily", "daily_basic", "moneyflow"]
        trade_dates = self._trade_dates(start_date, end_date)
        if not trade_dates:
            raise DataSyncError(f"No open trading days returned by Tushare trade_cal for {start_date}-{end_date}.")

        totals = {name: 0 for name in selected}
        failures: list[str] = []
        for idx, trade_date in enumerate(trade_dates, start=1):
            day_result: dict[str, int | str] = {"trade_date": trade_date}
            for dataset in selected:
                try:
                    rows = self._sync_one_dataset_by_trade_date(dataset, trade_date)
                    totals[dataset] += rows
                    day_result[dataset] = rows
                except Exception as exc:
                    failures.append(f"{trade_date}:{dataset}:{exc}")
                    day_result[dataset] = "failed"
            self._record_run(
                "calendar_batch_day",
                "partial_success" if any(value == "failed" for value in day_result.values()) else "success",
                trade_date=trade_date,
                affected_rows=sum(value for value in day_result.values() if isinstance(value, int)),
                message=str(day_result),
            )
            print(
                f"[calendar_batch] {idx}/{len(trade_dates)} {trade_date} "
                + " ".join(f"{name}={totals.get(name, 0)}" for name in selected)
                + f" failed={len(failures)}",
                flush=True,
            )

        self._record_run(
            "calendar_batch",
            "partial_success" if failures else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=sum(totals.values()),
            message=";".join(failures[:30]) if failures else None,
        )
        return {
            "task": "calendar_batch",
            "start_date": start_date,
            "end_date": end_date,
            "trade_date_count": len(trade_dates),
            "datasets": selected,
            "affected_rows": totals,
            "failed_count": len(failures),
            "failed_preview": failures[:10],
        }

    def _sync_one_dataset_by_trade_date(self, dataset: str, trade_date: str) -> int:
        if dataset == "daily":
            df = self.client.get_daily_quotes(trade_date=trade_date)
            return self._upsert_daily_quotes(df) if not df.empty else 0
        if dataset == "daily_basic":
            df = self.client.get_daily_basic(trade_date=trade_date)
            return self._upsert_daily_basic(df) if not df.empty else 0
        if dataset == "moneyflow":
            df = self.client.get_moneyflow(trade_date=trade_date)
            return self._upsert_moneyflow(df) if not df.empty else 0
        if dataset == "adj_factor":
            df = self.client.get_adj_factor(trade_date=trade_date)
            return self._upsert_adj_factor(df) if not df.empty else 0
        if dataset == "stk_limit":
            df = self.client.get_stk_limit(trade_date=trade_date)
            return self._upsert_stk_limit(df) if not df.empty else 0
        if dataset == "index_daily":
            total = 0
            for index_code in CORE_INDEX_CODES:
                df = self.client.get_index_daily(ts_code=index_code, start_date=trade_date, end_date=trade_date)
                total += self._upsert_index_daily(df) if not df.empty else 0
            return total
        raise DataSyncError(f"Unsupported calendar batch dataset: {dataset}")

    def _trade_dates(self, start_date: str, end_date: str) -> list[str]:
        df = self.client.get_trade_calendar(start_date=start_date, end_date=end_date)
        if df.empty or "cal_date" not in df.columns:
            return []
        return sorted(df["cal_date"].astype(str).tolist())

    def sync_daily_history_full_market(self, start_date: str, end_date: str) -> dict:
        stock_pool = self.client.get_full_stock_universe()
        if stock_pool.empty:
            raise DataSyncError("Tushare returned an empty full-market stock pool.")
        total_rows = 0
        failed: list[str] = []
        for idx, ts_code in enumerate(stock_pool["ts_code"].tolist(), start=1):
            try:
                df = self.client.get_daily_quotes(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                total_rows += self._upsert_daily_quotes(df)
            except Exception as exc:
                failed.append(f"{ts_code}:{exc}")
            if idx % 100 == 0:
                print(f"[daily_history] processed={idx}/{len(stock_pool)} rows={total_rows} failed={len(failed)}", flush=True)
        self._record_run(
            "daily_history_full_market",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=";".join(failed[:20]) if failed else None,
        )
        return {
            "task": "daily_history_full_market",
            "start_date": start_date,
            "end_date": end_date,
            "full_market_count": len(stock_pool),
            "affected_rows": total_rows,
            "failed_count": len(failed),
        }

    def sync_daily_basic_history_full_market(self, start_date: str, end_date: str) -> dict:
        stock_pool = self.client.get_full_stock_universe()
        if stock_pool.empty:
            raise DataSyncError("Tushare returned an empty full-market stock pool.")
        total_rows = 0
        failed: list[str] = []
        for idx, ts_code in enumerate(stock_pool["ts_code"].tolist(), start=1):
            try:
                df = self.client.get_daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                total_rows += self._upsert_daily_basic(df)
            except Exception as exc:
                failed.append(f"{ts_code}:{exc}")
            if idx % 100 == 0:
                print(f"[daily_basic_history] processed={idx}/{len(stock_pool)} rows={total_rows} failed={len(failed)}", flush=True)
        self._record_run(
            "daily_basic_history_full_market",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=";".join(failed[:20]) if failed else None,
        )
        return {
            "task": "daily_basic_history_full_market",
            "start_date": start_date,
            "end_date": end_date,
            "full_market_count": len(stock_pool),
            "affected_rows": total_rows,
            "failed_count": len(failed),
        }

    def sync_moneyflow_full_market(self, start_date: str, end_date: str) -> dict:
        stock_pool = self.client.get_full_stock_universe()
        if stock_pool.empty:
            raise DataSyncError("Tushare returned an empty full-market stock pool.")
        total_rows = 0
        empty_count = 0
        failed: list[str] = []
        for idx, ts_code in enumerate(stock_pool["ts_code"].tolist(), start=1):
            try:
                df = self.client.get_moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    empty_count += 1
                    continue
                total_rows += self._upsert_moneyflow(df)
            except Exception as exc:
                failed.append(f"{ts_code}:{exc}")
            if idx % 100 == 0:
                print(f"[moneyflow] processed={idx}/{len(stock_pool)} rows={total_rows} empty={empty_count} failed={len(failed)}", flush=True)
        self._record_run(
            "moneyflow_full_market",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=f"empty_stock_count={empty_count};failed={';'.join(failed[:20]) if failed else ''}",
        )
        return {
            "task": "moneyflow_full_market",
            "start_date": start_date,
            "end_date": end_date,
            "full_market_count": len(stock_pool),
            "empty_stock_count": empty_count,
            "affected_rows": total_rows,
            "failed_count": len(failed),
        }

    def sync_index_daily(self, start_date: str, end_date: str, index_codes: list[str] | None = None) -> dict:
        selected = index_codes or list(CORE_INDEX_CODES)
        totals: dict[str, int | str] = {}
        failed: list[str] = []
        for index_code in selected:
            try:
                df = self.client.get_index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                totals[index_code] = self._upsert_index_daily(df) if not df.empty else 0
            except Exception as exc:
                totals[index_code] = "failed"
                failed.append(f"{index_code}:{exc}")
        self._record_run(
            "index_daily",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=sum(value for value in totals.values() if isinstance(value, int)),
            message=";".join(failed[:20]) if failed else None,
        )
        return {"task": "index_daily", "start_date": start_date, "end_date": end_date, "indexes": totals, "failed_count": len(failed)}

    def sync_index_basic(self, markets: list[str] | None = None) -> dict:
        selected = markets or list(INDEX_BASIC_MARKETS)
        summary: dict[str, dict] = {}
        failed: list[str] = []
        total_rows = 0
        for market in selected:
            try:
                df = self.client.get_index_basic(market=market)
                endpoint = f"index_basic_{market.lower()}"
                rows = self._upsert_raw_records(endpoint, df) if not df.empty else 0
                summary[market] = {"endpoint": endpoint, "affected_rows": rows}
                total_rows += rows
            except Exception as exc:
                failed.append(f"{market}:{exc}")
                summary[market] = {"endpoint": f"index_basic_{market.lower()}", "affected_rows": 0, "error": str(exc)}
        self._record_run(
            "index_basic",
            "partial_success" if failed else "success",
            affected_rows=total_rows,
            message=json.dumps(summary, ensure_ascii=False)[:2000],
        )
        return {"task": "index_basic", "markets": summary, "affected_rows": total_rows, "failed_count": len(failed), "failed_preview": failed[:10]}

    def sync_index_daily_universe(
        self,
        start_date: str,
        end_date: str,
        markets: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        skip_complete: bool = True,
    ) -> dict:
        codes = self._index_codes_from_raw(markets=markets)
        bootstrap = None
        if not codes:
            bootstrap = self.sync_index_basic(markets=markets)
            codes = self._index_codes_from_raw(markets=markets)
        original_count = len(codes)
        expected_trade_days = len(self._trade_dates(start_date, end_date))
        skipped_complete = 0
        if skip_complete and expected_trade_days > 0:
            existing_counts = self._index_daily_date_counts(codes, start_date, end_date)
            before = len(codes)
            codes = [code for code in codes if existing_counts.get(code, 0) < expected_trade_days]
            skipped_complete = before - len(codes)
        if offset > 0:
            codes = codes[offset:]
        if limit is not None:
            codes = codes[:limit]
        if not codes:
            return {
                "task": "index_daily_universe",
                "start_date": start_date,
                "end_date": end_date,
                "index_count": 0,
                "original_index_count": original_count,
                "expected_trade_days": expected_trade_days,
                "skipped_complete": skipped_complete,
                "affected_rows": 0,
                "failed_count": 0,
                "failed_preview": [],
                "bootstrap_index_basic": bootstrap,
                "status": "nothing_to_sync",
            }

        totals: dict[str, int | str] = {}
        failed: list[str] = []
        for idx, index_code in enumerate(codes, start=1):
            try:
                df = self.client.get_index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                totals[index_code] = self._upsert_index_daily(df) if not df.empty else 0
            except Exception as exc:
                totals[index_code] = "failed"
                failed.append(f"{index_code}:{exc}")
            if idx % 100 == 0:
                print(f"[index_daily_universe] processed={idx}/{len(codes)} failed={len(failed)}", flush=True)
        affected = sum(value for value in totals.values() if isinstance(value, int))
        self._record_run(
            "index_daily_universe",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=affected,
            message=";".join(failed[:20]) if failed else None,
        )
        return {
            "task": "index_daily_universe",
            "start_date": start_date,
            "end_date": end_date,
            "index_count": len(codes),
            "original_index_count": original_count,
            "expected_trade_days": expected_trade_days,
            "skipped_complete": skipped_complete,
            "offset": offset,
            "affected_rows": affected,
            "failed_count": len(failed),
            "failed_preview": failed[:10],
            "bootstrap_index_basic": bootstrap,
        }

    def sync_stk_limit_batch(self, start_date: str, end_date: str) -> dict:
        trade_dates = self._trade_dates(start_date, end_date)
        total_rows = 0
        failed: list[str] = []
        for idx, trade_date in enumerate(trade_dates, start=1):
            try:
                df = self.client.get_stk_limit(trade_date=trade_date)
                total_rows += self._upsert_stk_limit(df) if not df.empty else 0
            except Exception as exc:
                failed.append(f"{trade_date}:{exc}")
            if idx % 20 == 0:
                print(f"[stk_limit] processed={idx}/{len(trade_dates)} rows={total_rows} failed={len(failed)}", flush=True)
        self._record_run(
            "stk_limit_batch",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=";".join(failed[:20]) if failed else None,
        )
        return {"task": "stk_limit_batch", "start_date": start_date, "end_date": end_date, "trade_date_count": len(trade_dates), "affected_rows": total_rows, "failed_count": len(failed)}

    def sync_adj_factor_full_market(self, start_date: str, end_date: str) -> dict:
        stock_pool = self.client.get_full_stock_universe()
        if stock_pool.empty:
            raise DataSyncError("Tushare returned an empty full-market stock pool.")
        total_rows = 0
        failed: list[str] = []
        for idx, ts_code in enumerate(stock_pool["ts_code"].tolist(), start=1):
            try:
                df = self.client.get_adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
                total_rows += self._upsert_adj_factor(df) if not df.empty else 0
            except Exception as exc:
                failed.append(f"{ts_code}:{exc}")
            if idx % 100 == 0:
                print(f"[adj_factor] processed={idx}/{len(stock_pool)} rows={total_rows} failed={len(failed)}", flush=True)
        self._record_run(
            "adj_factor_full_market",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=";".join(failed[:20]) if failed else None,
        )
        return {"task": "adj_factor_full_market", "start_date": start_date, "end_date": end_date, "full_market_count": len(stock_pool), "affected_rows": total_rows, "failed_count": len(failed)}

    def sync_financial_foundation(
        self,
        start_date: str,
        end_date: str,
        endpoints: list[str] | None = None,
        mode: str = "auto",
        offset: int = 0,
        limit: int | None = None,
        skip_existing: bool = True,
    ) -> dict:
        selected = endpoints or list(FINANCIAL_ENDPOINTS)
        invalid = sorted(set(selected) - set(FINANCIAL_ENDPOINTS))
        if invalid:
            raise DataSyncError(f"Unsupported financial endpoints: {invalid}")
        stock_pool = self.client.get_full_stock_universe()
        stock_codes = stock_pool["ts_code"].tolist() if not stock_pool.empty else []
        periods = self._quarter_periods(start_date, end_date)
        summary: dict[str, dict] = {}
        for endpoint in selected:
            if mode not in {"auto", "vip", "stock"}:
                raise DataSyncError("financial mode must be auto, vip, or stock")
            if mode in {"auto", "vip"} and endpoint in VIP_FINANCIAL_ENDPOINTS:
                vip_result = self._sync_financial_by_period(endpoint, periods)
                summary[f"{endpoint}_vip"] = vip_result
                if vip_result["affected_rows"] > 0 or mode == "vip":
                    continue
            if mode == "auto" and endpoint == "express":
                summary[endpoint] = self._sync_financial_by_range(endpoint, start_date, end_date)
                continue
            if mode == "auto" and endpoint == "forecast":
                summary[endpoint] = self._sync_forecast_by_ann_date(start_date, end_date)
                continue
            summary[endpoint] = self._sync_financial_by_stock(
                endpoint,
                stock_codes,
                start_date,
                end_date,
                offset=offset,
                limit=limit,
                skip_existing=skip_existing,
            )
        affected = sum(item["affected_rows"] for item in summary.values())
        failed = sum(item["failed_count"] for item in summary.values())
        self._record_run(
            "financial_foundation",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=affected,
            message=json.dumps({key: {"rows": value["affected_rows"], "failed": value["failed_count"]} for key, value in summary.items()}, ensure_ascii=False)[:2000],
        )
        return {"task": "financial_foundation", "start_date": start_date, "end_date": end_date, "mode": mode, "endpoints": summary, "affected_rows": affected, "failed_count": failed}

    def sync_v9_event_foundation(
        self,
        start_date: str,
        end_date: str,
        endpoints: list[str] | None = None,
        *,
        page_size: int = V9_EVENT_PAGE_SIZE,
        max_pages: int = V9_EVENT_MAX_PAGES,
    ) -> dict:
        selected = endpoints or list(V9_EVENT_ENDPOINTS)
        invalid = sorted(set(selected) - set(V9_EVENT_ENDPOINTS))
        if invalid:
            raise DataSyncError(f"Unsupported v9 event endpoints: {invalid}")
        if page_size < 1 or page_size > V9_EVENT_PAGE_SIZE:
            raise DataSyncError(f"v9 event page_size must be between 1 and {V9_EVENT_PAGE_SIZE}")
        if max_pages < 1:
            raise DataSyncError("v9 event max_pages must be positive")

        trade_dates = self._trade_dates(start_date, end_date) if set(selected) & V9_DAILY_EVENT_ENDPOINTS else []
        summary: dict[str, dict] = {}
        for endpoint in selected:
            if endpoint in V9_DAILY_EVENT_ENDPOINTS:
                result = self._sync_v9_daily_event_endpoint(endpoint, trade_dates)
            else:
                result = self._sync_v9_range_event_endpoint(
                    endpoint,
                    start_date,
                    end_date,
                    page_size=page_size,
                    max_pages=max_pages,
                )
            summary[endpoint] = result
            self._record_run(
                f"v9_event_{endpoint}",
                "partial_success" if result["failed_count"] else "success",
                start_date=start_date,
                end_date=end_date,
                affected_rows=result["affected_rows"],
                message=json.dumps(
                    {
                        "queries": result["query_count"],
                        "pages": result.get("page_count"),
                        "failed": result["failed_preview"],
                    },
                    ensure_ascii=False,
                )[:2000],
            )

        affected = sum(item["affected_rows"] for item in summary.values())
        failed = sum(item["failed_count"] for item in summary.values())
        self._record_run(
            "v9_event_foundation",
            "partial_success" if failed else "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=affected,
            message=json.dumps(
                {name: {"rows": item["affected_rows"], "failed": item["failed_count"]} for name, item in summary.items()},
                ensure_ascii=False,
            )[:2000],
        )
        return {
            "task": "v9_event_foundation",
            "start_date": start_date,
            "end_date": end_date,
            "data_source": "Tushare Pro",
            "endpoints": summary,
            "affected_rows": affected,
            "failed_count": failed,
            "diagnostic_only": True,
            "promotion_allowed": False,
            "policy": "Sparse real events are persisted as returned; empty successful queries remain empty and are never simulated.",
        }

    def _sync_v9_daily_event_endpoint(self, endpoint: str, trade_dates: list[str]) -> dict:
        affected = 0
        failures: list[str] = []
        for index, trade_date in enumerate(trade_dates, start=1):
            try:
                if endpoint == "top_list":
                    frame = self.client.get_top_list(trade_date)
                elif endpoint == "top_inst":
                    frame = self.client.get_top_inst(trade_date)
                elif endpoint == "limit_list_d":
                    frame = self.client.get_limit_list(trade_date)
                elif endpoint == "suspend_d":
                    frame = self.client.get_suspend_daily(trade_date)
                else:
                    raise DataSyncError(f"Unsupported daily v9 event endpoint: {endpoint}")
                affected += self._upsert_raw_records(endpoint, frame) if not frame.empty else 0
            except Exception as exc:
                failures.append(f"{trade_date}:{exc}")
            if index % 20 == 0 or index == len(trade_dates):
                print(
                    f"[v9_event:{endpoint}] processed={index}/{len(trade_dates)} rows={affected} failed={len(failures)}",
                    flush=True,
                )
        return {
            "endpoint": endpoint,
            "query_mode": "trade_date",
            "query_count": len(trade_dates),
            "affected_rows": affected,
            "failed_count": len(failures),
            "failed_preview": failures[:10],
        }

    def _sync_v9_range_event_endpoint(
        self,
        endpoint: str,
        start_date: str,
        end_date: str,
        *,
        page_size: int,
        max_pages: int,
    ) -> dict:
        affected = 0
        page_count = 0
        failures: list[str] = []
        pending_chunks = self._date_range_chunks(start_date, end_date, V9_EVENT_RANGE_CHUNK_DAYS)
        initial_chunk_count = len(pending_chunks)
        processed_chunk_count = 0
        split_count = 0
        all_exhausted = True
        while pending_chunks:
            chunk_start, chunk_end = pending_chunks.pop(0)
            buffered_frames: list[pd.DataFrame] = []
            split_range = False
            chunk_exhausted = False
            chunk_failed = False
            for page in range(max_pages):
                offset = page * page_size
                try:
                    frame = self.client.get_v9_event_range(
                        endpoint,
                        chunk_start,
                        chunk_end,
                        limit=page_size,
                        offset=offset,
                    )
                except Exception as exc:
                    failures.append(f"{chunk_start}-{chunk_end}:offset={offset}:{exc}")
                    chunk_failed = True
                    break
                page_count += 1
                if frame.empty:
                    chunk_exhausted = True
                    break
                buffered_frames.append(frame)
                print(
                    f"[v9_event:{endpoint}] range={chunk_start}-{chunk_end} page={page + 1} "
                    f"offset={offset} page_rows={len(frame)} buffered_rows={sum(len(item) for item in buffered_frames)} "
                    f"total_rows={affected}",
                    flush=True,
                )
                if len(frame) < page_size:
                    chunk_exhausted = True
                    break
                next_offset = offset + page_size
                if next_offset > V9_EVENT_MAX_OFFSET:
                    split_range = True
                    break
            if not chunk_exhausted and not chunk_failed and not split_range:
                split_range = True

            if split_range:
                split_chunks = self._split_date_range(chunk_start, chunk_end)
                if split_chunks is None:
                    all_exhausted = False
                    failures.append(f"{chunk_start}-{chunk_end}:pagination_not_exhausted_with_single_day_range")
                    continue
                split_count += 1
                pending_chunks = [*split_chunks, *pending_chunks]
                print(
                    f"[v9_event:{endpoint}] split_range={chunk_start}-{chunk_end} into "
                    f"{split_chunks[0][0]}-{split_chunks[0][1]} and {split_chunks[1][0]}-{split_chunks[1][1]}",
                    flush=True,
                )
                continue
            if chunk_failed:
                all_exhausted = False
                continue
            processed_chunk_count += 1
            for frame in buffered_frames:
                affected += self._upsert_raw_records(endpoint, frame)
        return {
            "endpoint": endpoint,
            "query_mode": "date_range_paginated",
            "query_count": page_count,
            "page_count": page_count,
            "chunk_count": processed_chunk_count,
            "initial_chunk_count": initial_chunk_count,
            "split_count": split_count,
            "chunk_days": V9_EVENT_RANGE_CHUNK_DAYS,
            "page_size": page_size,
            "pagination_exhausted": all_exhausted,
            "affected_rows": affected,
            "failed_count": len(failures),
            "failed_preview": failures[:10],
        }

    @staticmethod
    def _date_range_chunks(start_date: str, end_date: str, chunk_days: int) -> list[tuple[str, str]]:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        if start > end:
            raise DataSyncError("start_date must not be after end_date")
        chunks = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
            cursor = chunk_end + timedelta(days=1)
        return chunks

    @staticmethod
    def _split_date_range(start_date: str, end_date: str) -> list[tuple[str, str]] | None:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        if start >= end:
            return None
        midpoint = start + ((end - start) // 2)
        left = (start.strftime("%Y%m%d"), midpoint.strftime("%Y%m%d"))
        right_start = midpoint + timedelta(days=1)
        right = (right_start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        return [left, right]

    def v9_event_status(self, start_date: str, end_date: str) -> dict:
        with session_scope() as session:
            run_rows = (
                session.query(DataSyncRun)
                .filter(DataSyncRun.task_type.in_([f"v9_event_{name}" for name in V9_EVENT_ENDPOINTS]))
                .order_by(DataSyncRun.created_at.desc())
                .all()
            )
            endpoint_rows = {name: 0 for name in V9_EVENT_ENDPOINTS}
            endpoint_stocks: dict[str, set[str]] = {name: set() for name in V9_EVENT_ENDPOINTS}
            for endpoint in V9_EVENT_ENDPOINTS:
                date_expr = "COALESCE(payload ->> 'float_date', trade_date, ann_date)" if endpoint == "share_float" else "COALESCE(trade_date, ann_date)"
                params = {"endpoint": endpoint, "start_date": start_date, "end_date": end_date}
                endpoint_rows[endpoint] = int(
                    session.execute(
                        text(
                            f"""
                            SELECT COUNT(*)
                            FROM tushare_raw_records
                            WHERE endpoint = :endpoint
                              AND {date_expr} >= :start_date
                              AND {date_expr} <= :end_date
                            """
                        ),
                        params,
                    ).scalar_one()
                )
                codes = session.execute(
                    text(
                        f"""
                        SELECT DISTINCT ts_code
                        FROM tushare_raw_records
                        WHERE endpoint = :endpoint
                          AND ts_code IS NOT NULL
                          AND ts_code <> ''
                          AND {date_expr} >= :start_date
                          AND {date_expr} <= :end_date
                        """
                    ),
                    params,
                ).scalars()
                endpoint_stocks[endpoint] = set(codes)

        all_codes = set().union(*endpoint_stocks.values())
        non_a_share_codes = {code for code in all_codes if _asset_class_for_ts_code(code) != "a_share_like"}

        latest_runs: dict[str, DataSyncRun] = {}
        for run in run_rows:
            endpoint = run.task_type.removeprefix("v9_event_")
            if endpoint in latest_runs:
                continue
            if run.start_date and run.start_date <= start_date and run.end_date and run.end_date >= end_date:
                latest_runs[endpoint] = run
        endpoint_status = {
            endpoint: {
                "status": latest_runs[endpoint].status if endpoint in latest_runs else "not_synced",
                "rows": endpoint_rows[endpoint],
                "stock_count": len(endpoint_stocks[endpoint]),
                "last_run_at": latest_runs[endpoint].created_at.isoformat() if endpoint in latest_runs else None,
            }
            for endpoint in V9_EVENT_ENDPOINTS
        }
        blocking = [
            endpoint
            for endpoint, item in endpoint_status.items()
            if item["status"] != "success"
        ]
        return {
            "task": "v9_event_status",
            "start_date": start_date,
            "end_date": end_date,
            "status": "blocked" if blocking else "pass",
            "blocking_endpoints": blocking,
            "endpoints": endpoint_status,
            "non_a_share_raw_code_count": len(non_a_share_codes),
            "non_a_share_raw_code_preview": sorted(non_a_share_codes)[:20],
            "policy": "Sparse endpoints are gated by successful audited queries, not by fake universe coverage. Non-A-share raw rows never enter model features without the stock-universe filter.",
        }

    def migrate_legacy_raw_financial_records(
        self,
        endpoints: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        selected = endpoints or list(LEGACY_RAW_FINANCIAL_TABLES)
        invalid = sorted(set(selected) - set(LEGACY_RAW_FINANCIAL_TABLES))
        if invalid:
            raise DataSyncError(f"Unsupported legacy financial endpoints: {invalid}")

        summary: dict[str, dict] = {}
        total_rows = 0
        for endpoint in selected:
            table = LEGACY_RAW_FINANCIAL_TABLES[endpoint]
            if not self._table_exists(table):
                summary[endpoint] = {"table": table, "affected_rows": 0, "status": "missing_table"}
                continue
            df = self._legacy_table_frame(table, start_date=start_date, end_date=end_date)
            rows = self._upsert_raw_records(endpoint, df) if not df.empty else 0
            summary[endpoint] = {"table": table, "affected_rows": rows, "status": "migrated"}
            total_rows += rows

        self._record_run(
            "legacy_financial_raw_migration",
            "success",
            start_date=start_date,
            end_date=end_date,
            affected_rows=total_rows,
            message=json.dumps(summary, ensure_ascii=False)[:2000],
        )
        return {
            "task": "legacy_financial_raw_migration",
            "start_date": start_date,
            "end_date": end_date,
            "endpoints": summary,
            "affected_rows": total_rows,
            "policy": "Migrates legacy raw_* financial tables into tushare_raw_records without deleting source rows.",
        }

    def _sync_financial_by_period(self, endpoint: str, periods: list[str]) -> dict:
        endpoint_name = f"{endpoint}_vip"
        total_rows = 0
        failed: list[str] = []
        for idx, period in enumerate(periods, start=1):
            try:
                df = self.client.get_financial_endpoint(endpoint_name, period=period)
                total_rows += self._upsert_raw_records(endpoint_name, df) if not df.empty else 0
            except Exception as exc:
                failed.append(f"{period}:{exc}")
            if idx % 8 == 0:
                print(f"[{endpoint_name}] processed={idx}/{len(periods)} rows={total_rows} failed={len(failed)}", flush=True)
        return {"endpoint": endpoint_name, "mode": "period", "period_count": len(periods), "affected_rows": total_rows, "failed_count": len(failed), "failed_preview": failed[:10]}

    def _sync_financial_by_range(self, endpoint: str, start_date: str, end_date: str) -> dict:
        try:
            df = self.client.get_financial_endpoint(endpoint, start_date=start_date, end_date=end_date)
            rows = self._upsert_raw_records(endpoint, df) if not df.empty else 0
            failed = []
        except Exception as exc:
            rows = 0
            failed = [str(exc)]
        return {
            "endpoint": endpoint,
            "mode": "range",
            "start_date": start_date,
            "end_date": end_date,
            "affected_rows": rows,
            "failed_count": len(failed),
            "failed_preview": failed[:10],
        }

    def _sync_forecast_by_ann_date(self, start_date: str, end_date: str) -> dict:
        total_rows = 0
        failed: list[str] = []
        dates = self._calendar_dates(start_date, end_date)
        for idx, ann_date in enumerate(dates, start=1):
            try:
                df = self.client.get_financial_endpoint("forecast", ann_date=ann_date)
                total_rows += self._upsert_raw_records("forecast", df) if not df.empty else 0
            except Exception as exc:
                failed.append(f"{ann_date}:{exc}")
            if idx % 100 == 0:
                print(f"[forecast] processed={idx}/{len(dates)} rows={total_rows} failed={len(failed)}", flush=True)
        return {
            "endpoint": "forecast",
            "mode": "ann_date",
            "date_count": len(dates),
            "affected_rows": total_rows,
            "failed_count": len(failed),
            "failed_preview": failed[:10],
        }

    def _sync_financial_by_stock(
        self,
        endpoint: str,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        offset: int = 0,
        limit: int | None = None,
        skip_existing: bool = True,
    ) -> dict:
        total_rows = 0
        failed: list[str] = []
        if endpoint == "fina_mainbz":
            iterator = [(ts_code, item_type) for ts_code in stock_codes for item_type in FINA_MAINBZ_TYPES]
            endpoint_names = [f"{endpoint}_{item_type}" for item_type in FINA_MAINBZ_TYPES]
        else:
            iterator = [(ts_code, None) for ts_code in stock_codes]
            endpoint_names = [endpoint]
        original_request_count = len(iterator)
        existing_by_endpoint = self._existing_raw_ts_codes(endpoint_names, start_date, end_date) if skip_existing else {}
        remaining: list[tuple[str, str | None]] = []
        skipped_existing = 0
        for ts_code, item_type in iterator:
            endpoint_name = f"{endpoint}_{item_type}" if item_type else endpoint
            if skip_existing and ts_code in existing_by_endpoint.get(endpoint_name, set()):
                skipped_existing += 1
                continue
            remaining.append((ts_code, item_type))
        if offset > 0:
            remaining = remaining[offset:]
        if limit is not None:
            remaining = remaining[:limit]

        attempted = 0
        for idx, (ts_code, item_type) in enumerate(remaining, start=1):
            try:
                extra = {"type": item_type} if item_type else {}
                endpoint_name = f"{endpoint}_{item_type}" if item_type else endpoint
                attempted += 1
                df = self.client.get_financial_endpoint(endpoint, ts_code=ts_code, start_date=start_date, end_date=end_date, **extra)
                total_rows += self._upsert_raw_records(endpoint_name, df) if not df.empty else 0
            except Exception as exc:
                failed.append(f"{ts_code}:{item_type or ''}:{exc}")
            if idx % 300 == 0:
                print(f"[{endpoint}] scanned={idx}/{len(iterator)} attempted={attempted} rows={total_rows} skipped={skipped_existing} failed={len(failed)}", flush=True)
        return {
            "endpoint": endpoint,
            "mode": "stock",
            "request_count": attempted,
            "scanned_count": len(remaining),
            "original_request_count": original_request_count,
            "offset": offset,
            "limit": limit,
            "skip_existing": skip_existing,
            "skipped_existing": skipped_existing,
            "affected_rows": total_rows,
            "failed_count": len(failed),
            "failed_preview": failed[:10],
        }

    def data_status(self) -> dict:
        with session_scope() as session:
            stock_count = session.scalar(select(func.count()).select_from(Stock)) or 0
            quote_count = session.scalar(select(func.count()).select_from(DailyQuote)) or 0
            basic_count = session.scalar(select(func.count()).select_from(DailyBasic)) or 0
            moneyflow_count = session.scalar(select(func.count()).select_from(MoneyflowDaily)) or 0
            index_count = session.scalar(select(func.count()).select_from(IndexDaily)) or 0
            adj_factor_count = session.scalar(select(func.count()).select_from(StockAdjFactor)) or 0
            limit_count = session.scalar(select(func.count()).select_from(StockLimitDaily)) or 0
            raw_count = session.scalar(select(func.count()).select_from(TushareRawRecord)) or 0
            latest_quote_date = session.scalar(select(func.max(DailyQuote.trade_date)))
            latest_basic_date = session.scalar(select(func.max(DailyBasic.trade_date)))
            latest_moneyflow_date = session.scalar(select(func.max(MoneyflowDaily.trade_date)))
            latest_index_date = session.scalar(select(func.max(IndexDaily.trade_date)))
            latest_limit_date = session.scalar(select(func.max(StockLimitDaily.trade_date)))
            latest_adj_factor_date = session.scalar(select(func.max(StockAdjFactor.trade_date)))
            raw_endpoint_rows = (
                session.query(TushareRawRecord.endpoint, func.count(TushareRawRecord.id))
                .group_by(TushareRawRecord.endpoint)
                .order_by(TushareRawRecord.endpoint.asc())
                .all()
            )
            last_run = (
                session.query(DataSyncRun)
                .order_by(DataSyncRun.created_at.desc())
                .limit(1)
                .one_or_none()
            )
            return {
                "stock_count": stock_count,
                "daily_quote_count": quote_count,
                "daily_basic_count": basic_count,
                "moneyflow_count": moneyflow_count,
                "index_daily_count": index_count,
                "adj_factor_count": adj_factor_count,
                "stock_limit_count": limit_count,
                "tushare_raw_record_count": raw_count,
                "latest_quote_date": latest_quote_date,
                "latest_basic_date": latest_basic_date,
                "latest_moneyflow_date": latest_moneyflow_date,
                "latest_index_date": latest_index_date,
                "latest_limit_date": latest_limit_date,
                "latest_adj_factor_date": latest_adj_factor_date,
                "raw_endpoint_rows": {row[0]: int(row[1]) for row in raw_endpoint_rows},
                "last_sync": None
                if last_run is None
                else {
                    "task_type": last_run.task_type,
                    "status": last_run.status,
                    "trade_date": last_run.trade_date,
                    "affected_rows": last_run.affected_rows,
                    "message": last_run.message,
                    "created_at": last_run.created_at.isoformat(),
                },
            }

    def stock_universe_audit(self, preview_limit: int = 20) -> dict:
        with session_scope() as session:
            total = int(session.scalar(select(func.count()).select_from(Stock)) or 0)
            active_count = int(
                session.scalar(select(func.count()).select_from(Stock).where(Stock.list_status == "L")) or 0
            )
            status_rows = (
                session.query(Stock.list_status, func.count(Stock.ts_code))
                .group_by(Stock.list_status)
                .order_by(Stock.list_status.asc())
                .all()
            )
            unknown_query = (
                session.query(Stock)
                .filter((Stock.list_status == "U") | ((Stock.name.like("UNKNOWN_%")) & (Stock.list_status != "N")))
                .order_by(Stock.ts_code.asc())
            )
            unknown_count = int(unknown_query.count())
            unknown_preview = unknown_query.limit(preview_limit).all()
            quarantined_unknown_count = int(
                session.query(Stock)
                .filter(Stock.list_status == "N", Stock.name.like("UNKNOWN_%"))
                .count()
            )
            unknown_codes = [
                row[0]
                for row in session.query(Stock.ts_code)
                .filter((Stock.list_status == "U") | ((Stock.name.like("UNKNOWN_%")) & (Stock.list_status != "N")))
                .all()
            ]

        blocking = []
        unknown_class_counts: dict[str, int] = {}
        for code in unknown_codes:
            asset_class = _asset_class_for_ts_code(code)
            unknown_class_counts[asset_class] = unknown_class_counts.get(asset_class, 0) + 1
        quarantined_classes = {"fund_or_index_like", "non_a_share_equity"}
        unresolved_unknown_count = sum(count for asset_class, count in unknown_class_counts.items() if asset_class not in quarantined_classes)
        if unresolved_unknown_count:
            blocking.append("unknown_stock_placeholders")
        if active_count < 1000:
            blocking.append("listed_stock_count_too_low")

        return {
            "task": "stock_universe_audit",
            "status": "blocked" if blocking else "pass",
            "blocking": blocking,
            "total_rows": total,
            "listed_stock_count": active_count,
            "list_status_counts": {str(status or "NULL"): int(count) for status, count in status_rows},
            "unknown_placeholder_count": unknown_count,
            "quarantined_unknown_placeholder_count": quarantined_unknown_count,
            "unknown_placeholder_class_counts": unknown_class_counts,
            "unresolved_unknown_placeholder_count": unresolved_unknown_count,
            "unknown_placeholder_preview": [
                {
                    "ts_code": row.ts_code,
                    "symbol": row.symbol,
                    "name": row.name,
                    "list_status": row.list_status,
                    "industry": row.industry,
                    "market": row.market,
                    "asset_class": _asset_class_for_ts_code(row.ts_code),
                }
                for row in unknown_preview
            ],
            "policy": (
                "UNKNOWN_* placeholders that look like funds or indexes are excluded from the A-share training universe. "
                "Unresolved placeholders remain blocking until repaired from Tushare stock_basic or explicitly quarantined."
            ),
        }

    def trade_date_snapshot_status(self, trade_date: str, *, min_cross_section_coverage: float = 0.90) -> dict:
        """Check that every v8 live feature dependency is present for one trading day."""
        with session_scope() as session:
            row_counts = {
                "daily_quotes": int(
                    session.scalar(select(func.count()).select_from(DailyQuote).where(DailyQuote.trade_date == trade_date)) or 0
                ),
                "daily_basic": int(
                    session.scalar(select(func.count()).select_from(DailyBasic).where(DailyBasic.trade_date == trade_date)) or 0
                ),
                "moneyflow": int(
                    session.scalar(select(func.count()).select_from(MoneyflowDaily).where(MoneyflowDaily.trade_date == trade_date)) or 0
                ),
                "stock_limit": int(
                    session.scalar(select(func.count()).select_from(StockLimitDaily).where(StockLimitDaily.trade_date == trade_date)) or 0
                ),
                "adj_factor": int(
                    session.scalar(select(func.count()).select_from(StockAdjFactor).where(StockAdjFactor.trade_date == trade_date)) or 0
                ),
                "index_daily": int(
                    session.scalar(
                        select(func.count(func.distinct(IndexDaily.ts_code)))
                        .select_from(IndexDaily)
                        .where(IndexDaily.trade_date == trade_date, IndexDaily.ts_code.in_(CORE_INDEX_CODES))
                    )
                    or 0
                ),
            }

        quote_rows = row_counts["daily_quotes"]
        coverage_vs_quotes = {
            name: self._coverage_ratio(row_counts[name], quote_rows)
            for name in ("daily_basic", "moneyflow", "stock_limit", "adj_factor")
        }
        blocking: list[str] = []
        if quote_rows < 1_000:
            blocking.append("daily_quotes_cross_section_too_small")
        blocking.extend(
            f"{name}_coverage_below_{min_cross_section_coverage:.0%}"
            for name, coverage in coverage_vs_quotes.items()
            if coverage is None or coverage < min_cross_section_coverage
        )
        missing_indexes = max(len(CORE_INDEX_CODES) - row_counts["index_daily"], 0)
        if missing_indexes:
            blocking.append("core_index_daily_incomplete")

        return {
            "task": "trade_date_snapshot_status",
            "trade_date": trade_date,
            "status": "blocked" if blocking else "pass",
            "blocking": blocking,
            "row_counts": row_counts,
            "coverage_vs_daily_quotes": coverage_vs_quotes,
            "min_cross_section_coverage": min_cross_section_coverage,
            "required_core_index_codes": list(CORE_INDEX_CODES),
            "missing_core_index_count": missing_indexes,
            "policy": "v8 scoring is blocked unless all real Tushare daily dependencies are present for the signal date.",
        }

    def training_frame_universe_audit(self, training_frame_path: str | Path, preview_limit: int = 20) -> dict:
        path = Path(training_frame_path)
        if not path.exists():
            raise DataSyncError(f"Training frame not found: {path}")

        codes: set[str] = set()
        rows = 0
        for chunk in pd.read_csv(path, usecols=["ts_code"], chunksize=200_000):
            rows += len(chunk)
            codes.update(str(code).strip() for code in chunk["ts_code"].dropna().tolist() if str(code).strip())

        with session_scope() as session:
            stock_rows = (
                session.query(Stock)
                .filter(Stock.ts_code.in_(sorted(codes)))
                .all()
                if codes
                else []
            )
        stocks = {row.ts_code: row for row in stock_rows}
        missing = sorted(codes - set(stocks))
        unknown = sorted(
            code
            for code, row in stocks.items()
            if row.list_status == "U" or str(row.name or "").startswith("UNKNOWN_")
        )
        quarantined = sorted(code for code, row in stocks.items() if row.list_status == "N")
        unsupported_status = sorted(code for code, row in stocks.items() if row.list_status not in {"L", "D"})
        non_a_share_like = sorted(code for code in codes if _asset_class_for_ts_code(code) != "a_share_like")

        issues = []
        if missing:
            issues.append("missing_stock_metadata")
        if unknown:
            issues.append("unknown_stock_metadata")
        if quarantined:
            issues.append("quarantined_non_stock_assets")
        if unsupported_status:
            issues.append("unsupported_stock_status")
        if non_a_share_like:
            issues.append("non_a_share_codes")

        return {
            "task": "training_frame_universe_audit",
            "training_frame_path": str(path),
            "rows": rows,
            "unique_codes": len(codes),
            "status": "pass" if not issues else "blocked",
            "issues": issues,
            "missing_stock_metadata_count": len(missing),
            "unknown_stock_metadata_count": len(unknown),
            "quarantined_non_stock_asset_count": len(quarantined),
            "unsupported_stock_status_count": len(unsupported_status),
            "non_a_share_code_count": len(non_a_share_like),
            "missing_stock_metadata_preview": missing[:preview_limit],
            "unknown_stock_metadata_preview": unknown[:preview_limit],
            "quarantined_non_stock_asset_preview": quarantined[:preview_limit],
            "unsupported_stock_status_preview": unsupported_status[:preview_limit],
            "non_a_share_code_preview": non_a_share_like[:preview_limit],
            "policy": "Training frames may include listed or delisted A-share stocks only; quarantined assets and UNKNOWN metadata are blocked.",
        }

    def quarantine_non_stock_placeholders(self, dry_run: bool = True, preview_limit: int = 20) -> dict:
        with session_scope() as session:
            placeholders = (
                session.query(Stock)
                .filter((Stock.list_status == "U") | (Stock.name.like("UNKNOWN_%")))
                .order_by(Stock.ts_code.asc())
                .all()
            )
            quarantined_classes = {"fund_or_index_like", "non_a_share_equity"}
            fund_or_index = [row for row in placeholders if _asset_class_for_ts_code(row.ts_code) in quarantined_classes]
            unresolved = [row for row in placeholders if _asset_class_for_ts_code(row.ts_code) not in quarantined_classes]
            if not dry_run:
                now = datetime.utcnow()
                for row in fund_or_index:
                    row.list_status = "N"
                    row.industry = row.industry or "NON_STOCK_ASSET"
                    row.market = row.market or "NON_A_SHARE"
                    row.updated_at = now

        if not dry_run:
            self._record_run(
                "quarantine_non_stock_placeholders",
                "success",
                affected_rows=len(fund_or_index),
                message=f"unresolved_unknown_placeholders={len(unresolved)}",
            )
        return {
            "task": "quarantine_non_stock_placeholders",
            "dry_run": dry_run,
            "quarantinable_count": len(fund_or_index),
            "unresolved_count": len(unresolved),
            "quarantinable_preview": [
                {"ts_code": row.ts_code, "name": row.name, "asset_class": _asset_class_for_ts_code(row.ts_code)}
                for row in fund_or_index[:preview_limit]
            ],
            "unresolved_preview": [
                {"ts_code": row.ts_code, "name": row.name, "asset_class": _asset_class_for_ts_code(row.ts_code)}
                for row in unresolved[:preview_limit]
            ],
            "policy": "Dry run by default. Use --apply only after checking the preview.",
        }

    def repair_unknown_a_share_metadata(self, dry_run: bool = True, limit: int | None = None) -> dict:
        with session_scope() as session:
            unknown_codes = [
                row[0]
                for row in session.query(Stock.ts_code)
                .filter((Stock.list_status == "U") | ((Stock.name.like("UNKNOWN_%")) & (Stock.list_status != "N")))
                .order_by(Stock.ts_code.asc())
                .all()
                if _asset_class_for_ts_code(row[0]) == "a_share_like"
            ]
        if limit is not None:
            unknown_codes = unknown_codes[:limit]

        repaired: list[dict] = []
        unresolved: list[dict] = []
        for ts_code in unknown_codes:
            metadata = self._lookup_auxiliary_stock_metadata(ts_code)
            if not metadata:
                unresolved.append({"ts_code": ts_code, "reason": "no_auxiliary_metadata"})
                continue
            repaired.append({"ts_code": ts_code, **metadata})

        if repaired and not dry_run:
            now = datetime.utcnow()
            with session_scope() as session:
                for item in repaired:
                    stock = session.get(Stock, item["ts_code"])
                    if stock is None:
                        continue
                    stock.symbol = stock.symbol or item["ts_code"].split(".")[0]
                    stock.name = item["name"]
                    stock.area = item.get("area")
                    stock.industry = item.get("industry")
                    stock.market = item.get("market") or stock.market
                    stock.list_date = item.get("list_date")
                    stock.list_status = item.get("list_status") or "D"
                    stock.updated_at = now
            self._record_run(
                "repair_unknown_a_share_metadata",
                "success",
                affected_rows=len(repaired),
                message=json.dumps({"unresolved": unresolved[:20]}, ensure_ascii=False)[:2000],
            )

        return {
            "task": "repair_unknown_a_share_metadata",
            "dry_run": dry_run,
            "candidate_count": len(unknown_codes),
            "repaired_count": len(repaired),
            "unresolved_count": len(unresolved),
            "repaired_preview": repaired[:20],
            "unresolved_preview": unresolved[:20],
            "policy": "Uses Tushare auxiliary endpoints only; it does not fabricate stock names.",
        }

    def foundation_status(self, start_date: str, end_date: str) -> dict:
        with session_scope() as session:
            quote_days = self._date_count(session, DailyQuote, start_date, end_date)
            basic_days = self._date_count(session, DailyBasic, start_date, end_date)
            moneyflow_days = self._date_count(session, MoneyflowDaily, start_date, end_date)
            index_days = self._date_count(session, IndexDaily, start_date, end_date)
            limit_days = self._date_count(session, StockLimitDaily, start_date, end_date)
            adj_factor_days = self._date_count(session, StockAdjFactor, start_date, end_date)
            total_trade_days = quote_days
            raw_endpoint_rows = {
                row[0]: int(row[1])
                for row in session.query(TushareRawRecord.endpoint, func.count(TushareRawRecord.id))
                .filter(
                    ((TushareRawRecord.ann_date >= start_date) & (TushareRawRecord.ann_date <= end_date))
                    | ((TushareRawRecord.end_date >= start_date) & (TushareRawRecord.end_date <= end_date))
                    | ((TushareRawRecord.period >= start_date) & (TushareRawRecord.period <= end_date))
                )
                .group_by(TushareRawRecord.endpoint)
                .all()
            }
            raw_stock_universe_codes = {
                str(row[0])
                for row in session.query(DailyQuote.ts_code)
                .join(Stock, Stock.ts_code == DailyQuote.ts_code)
                .filter(
                    DailyQuote.trade_date >= start_date,
                    DailyQuote.trade_date <= end_date,
                    Stock.list_status.in_(["L", "D"]),
                )
                .distinct()
                .all()
            }
            raw_stock_universe_count = len(raw_stock_universe_codes)
            raw_endpoint_stock_counts = {
                row[0]: int(row[1])
                for row in session.query(TushareRawRecord.endpoint, func.count(func.distinct(TushareRawRecord.ts_code)))
                .filter(
                    TushareRawRecord.ts_code.isnot(None),
                    (
                        ((TushareRawRecord.ann_date >= start_date) & (TushareRawRecord.ann_date <= end_date))
                        | ((TushareRawRecord.end_date >= start_date) & (TushareRawRecord.end_date <= end_date))
                        | ((TushareRawRecord.period >= start_date) & (TushareRawRecord.period <= end_date))
                    ),
                )
                .group_by(TushareRawRecord.endpoint)
                .all()
            }
            financial_field_rows = (
                session.query(TushareRawRecord.ts_code, TushareRawRecord.payload)
                .filter(
                    TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip")),
                    TushareRawRecord.ts_code.isnot(None),
                    (
                        ((TushareRawRecord.ann_date >= start_date) & (TushareRawRecord.ann_date <= end_date))
                        | ((TushareRawRecord.end_date >= start_date) & (TushareRawRecord.end_date <= end_date))
                        | ((TushareRawRecord.period >= start_date) & (TushareRawRecord.period <= end_date))
                    ),
                )
                .all()
            )
        present_raw = {endpoint for endpoint, rows in raw_endpoint_rows.items() if rows > 0}
        missing_raw = []
        for endpoint in FINANCIAL_ENDPOINTS:
            accepted = {endpoint}
            if endpoint in VIP_FINANCIAL_ENDPOINTS:
                accepted.add(f"{endpoint}_vip")
            if endpoint == "fina_mainbz":
                missing_mainbz = sorted({"fina_mainbz_P", "fina_mainbz_D", "fina_mainbz_I"} - present_raw)
                if missing_mainbz:
                    missing_raw.extend(missing_mainbz)
                continue
            if not (accepted & present_raw):
                missing_raw.append(endpoint)
        raw_endpoint_stock_coverage = {
            endpoint: self._coverage_ratio(raw_endpoint_stock_counts.get(endpoint, 0), raw_stock_universe_count)
            for endpoint in RAW_COVERAGE_REQUIRED_ENDPOINTS
        }
        financial_field_stock_counts = _financial_field_stock_counts(
            financial_field_rows,
            raw_stock_universe_codes,
        )
        financial_field_stock_coverage = {
            name: self._coverage_ratio(financial_field_stock_counts.get(name, 0), raw_stock_universe_count)
            for name in FINA_INDICATOR_FACTOR_NAMES
        }
        low_financial_field_coverage = [
            name
            for name in FINA_INDICATOR_GATE_FIELDS
            if financial_field_stock_coverage.get(name) is None
            or financial_field_stock_coverage[name] < MIN_FINANCIAL_FIELD_STOCK_COVERAGE
        ]
        low_raw_stock_coverage = [
            endpoint
            for endpoint, ratio in raw_endpoint_stock_coverage.items()
            if ratio is None or ratio < MIN_RAW_STOCK_COVERAGE
        ]
        coverage = {
            "daily_quotes": self._coverage_ratio(quote_days, total_trade_days),
            "daily_basic": self._coverage_ratio(basic_days, total_trade_days),
            "moneyflow": self._coverage_ratio(moneyflow_days, total_trade_days),
            "index_daily": self._coverage_ratio(index_days, total_trade_days),
            "stock_limit": self._coverage_ratio(limit_days, total_trade_days),
            "adj_factor": self._coverage_ratio(adj_factor_days, total_trade_days),
        }
        blocking = [
            name
            for name in ("daily_quotes", "daily_basic", "index_daily", "stock_limit", "adj_factor")
            if coverage[name] is None or coverage[name] < 0.95
        ]
        if missing_raw or low_raw_stock_coverage:
            blocking.append("financial_raw_records")
        if low_financial_field_coverage:
            blocking.append("financial_field_coverage")
        return {
            "task": "foundation_status",
            "start_date": start_date,
            "end_date": end_date,
            "trade_days": total_trade_days,
            "date_counts": {
                "daily_quotes": quote_days,
                "daily_basic": basic_days,
                "moneyflow": moneyflow_days,
                "index_daily": index_days,
                "stock_limit": limit_days,
                "adj_factor": adj_factor_days,
            },
            "coverage": coverage,
            "raw_endpoint_rows": raw_endpoint_rows,
            "raw_stock_universe_count": raw_stock_universe_count,
            "raw_endpoint_stock_counts": raw_endpoint_stock_counts,
            "raw_endpoint_stock_coverage": raw_endpoint_stock_coverage,
            "low_raw_stock_coverage": low_raw_stock_coverage,
            "min_raw_stock_coverage": MIN_RAW_STOCK_COVERAGE,
            "financial_field_stock_counts": financial_field_stock_counts,
            "financial_field_stock_coverage": financial_field_stock_coverage,
            "low_financial_field_coverage": low_financial_field_coverage,
            "min_financial_field_stock_coverage": MIN_FINANCIAL_FIELD_STOCK_COVERAGE,
            "structurally_sparse_financial_fields": list(FINA_INDICATOR_STRUCTURALLY_SPARSE_FIELDS),
            "missing_financial_endpoints": missing_raw,
            "status": "pass" if not blocking else "blocked",
            "blocking": blocking,
            "policy": "Do not export GPU training data until this status is pass or the user explicitly accepts a listed missing endpoint.",
        }

    @staticmethod
    def _date_count(session, model, start_date: str, end_date: str) -> int:
        return int(
            session.query(model.trade_date)
            .filter(model.trade_date >= start_date, model.trade_date <= end_date)
            .group_by(model.trade_date)
            .count()
        )

    @staticmethod
    def _coverage_ratio(count: int, total: int) -> float | None:
        if total <= 0:
            return None
        return round(min(count / total, 1.0), 6)

    def search_stocks(self, keyword: str, limit: int = 20) -> list[dict]:
        pattern = f"%{keyword.strip()}%"
        if not keyword.strip():
            return []
        with session_scope() as session:
            rows = (
                session.query(Stock)
                .filter((Stock.ts_code.ilike(pattern)) | (Stock.symbol.ilike(pattern)) | (Stock.name.ilike(pattern)))
                .order_by(Stock.ts_code.asc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "ts_code": row.ts_code,
                    "symbol": row.symbol,
                    "name": row.name,
                    "industry": row.industry,
                    "market": row.market,
                    "list_date": row.list_date,
                }
                for row in rows
            ]

    def _upsert_stocks(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        affected = 0
        with session_scope() as session:
            for row in df.to_dict("records"):
                stmt = insert(Stock).values(
                    ts_code=row.get("ts_code"),
                    symbol=row.get("symbol"),
                    name=row.get("name"),
                    area=row.get("area"),
                    industry=row.get("industry"),
                    market=row.get("market"),
                    list_date=row.get("list_date"),
                    list_status=row.get("list_status") or "L",
                    is_hs=row.get("is_hs"),
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Stock.ts_code],
                    set_={
                        "symbol": stmt.excluded.symbol,
                        "name": stmt.excluded.name,
                        "area": stmt.excluded.area,
                        "industry": stmt.excluded.industry,
                        "market": stmt.excluded.market,
                        "list_date": stmt.excluded.list_date,
                        "list_status": stmt.excluded.list_status,
                        "is_hs": stmt.excluded.is_hs,
                        "updated_at": now,
                    },
                )
                session.execute(stmt)
                affected += 1
        return affected

    def _upsert_daily_quotes(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "open": _clean_float(row.get("open")),
                "high": _clean_float(row.get("high")),
                "low": _clean_float(row.get("low")),
                "close": _clean_float(row.get("close")),
                "pre_close": _clean_float(row.get("pre_close")),
                "change": _clean_float(row.get("change")),
                "pct_chg": _clean_float(row.get("pct_chg")),
                "vol": _clean_float(row.get("vol")),
                "amount": _clean_float(row.get("amount")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            self._ensure_stock_placeholders(session, df)
            stmt = insert(DailyQuote)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_daily_quote_stock_date",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "pre_close": stmt.excluded.pre_close,
                    "change": stmt.excluded.change,
                    "pct_chg": stmt.excluded.pct_chg,
                    "vol": stmt.excluded.vol,
                    "amount": stmt.excluded.amount,
                    "updated_at": now,
                },
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_daily_basic(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "close": _clean_float(row.get("close")),
                "turnover_rate": _clean_float(row.get("turnover_rate")),
                "turnover_rate_f": _clean_float(row.get("turnover_rate_f")),
                "volume_ratio": _clean_float(row.get("volume_ratio")),
                "pe": _clean_float(row.get("pe")),
                "pe_ttm": _clean_float(row.get("pe_ttm")),
                "pb": _clean_float(row.get("pb")),
                "ps": _clean_float(row.get("ps")),
                "ps_ttm": _clean_float(row.get("ps_ttm")),
                "dv_ratio": _clean_float(row.get("dv_ratio")),
                "dv_ttm": _clean_float(row.get("dv_ttm")),
                "total_share": _clean_float(row.get("total_share")),
                "float_share": _clean_float(row.get("float_share")),
                "free_share": _clean_float(row.get("free_share")),
                "total_mv": _clean_float(row.get("total_mv")),
                "circ_mv": _clean_float(row.get("circ_mv")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            self._ensure_stock_placeholders(session, df)
            stmt = insert(DailyBasic)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_daily_basic_stock_date",
                set_={
                    "close": stmt.excluded.close,
                    "turnover_rate": stmt.excluded.turnover_rate,
                    "turnover_rate_f": stmt.excluded.turnover_rate_f,
                    "volume_ratio": stmt.excluded.volume_ratio,
                    "pe": stmt.excluded.pe,
                    "pe_ttm": stmt.excluded.pe_ttm,
                    "pb": stmt.excluded.pb,
                    "ps": stmt.excluded.ps,
                    "ps_ttm": stmt.excluded.ps_ttm,
                    "dv_ratio": stmt.excluded.dv_ratio,
                    "dv_ttm": stmt.excluded.dv_ttm,
                    "total_share": stmt.excluded.total_share,
                    "float_share": stmt.excluded.float_share,
                    "free_share": stmt.excluded.free_share,
                    "total_mv": stmt.excluded.total_mv,
                    "circ_mv": stmt.excluded.circ_mv,
                    "updated_at": now,
                },
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_moneyflow(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "buy_sm_amount": _clean_float(row.get("buy_sm_amount")),
                "sell_sm_amount": _clean_float(row.get("sell_sm_amount")),
                "buy_md_amount": _clean_float(row.get("buy_md_amount")),
                "sell_md_amount": _clean_float(row.get("sell_md_amount")),
                "buy_lg_amount": _clean_float(row.get("buy_lg_amount")),
                "sell_lg_amount": _clean_float(row.get("sell_lg_amount")),
                "buy_elg_amount": _clean_float(row.get("buy_elg_amount")),
                "sell_elg_amount": _clean_float(row.get("sell_elg_amount")),
                "net_mf_amount": _clean_float(row.get("net_mf_amount")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            self._ensure_stock_placeholders(session, df)
            stmt = insert(MoneyflowDaily)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_moneyflow_stock_date",
                set_={
                    "buy_sm_amount": stmt.excluded.buy_sm_amount,
                    "sell_sm_amount": stmt.excluded.sell_sm_amount,
                    "buy_md_amount": stmt.excluded.buy_md_amount,
                    "sell_md_amount": stmt.excluded.sell_md_amount,
                    "buy_lg_amount": stmt.excluded.buy_lg_amount,
                    "sell_lg_amount": stmt.excluded.sell_lg_amount,
                    "buy_elg_amount": stmt.excluded.buy_elg_amount,
                    "sell_elg_amount": stmt.excluded.sell_elg_amount,
                    "net_mf_amount": stmt.excluded.net_mf_amount,
                    "updated_at": now,
                },
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_index_daily(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "open": _clean_float(row.get("open")),
                "high": _clean_float(row.get("high")),
                "low": _clean_float(row.get("low")),
                "close": _clean_float(row.get("close")),
                "pre_close": _clean_float(row.get("pre_close")),
                "change": _clean_float(row.get("change")),
                "pct_chg": _clean_float(row.get("pct_chg")),
                "vol": _clean_float(row.get("vol")),
                "amount": _clean_float(row.get("amount")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            stmt = insert(IndexDaily)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_index_daily_code_date",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "pre_close": stmt.excluded.pre_close,
                    "change": stmt.excluded.change,
                    "pct_chg": stmt.excluded.pct_chg,
                    "vol": stmt.excluded.vol,
                    "amount": stmt.excluded.amount,
                    "updated_at": now,
                },
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_adj_factor(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "adj_factor": _clean_float(row.get("adj_factor")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            self._ensure_stock_placeholders(session, df)
            stmt = insert(StockAdjFactor)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_adj_factor_stock_date",
                set_={"adj_factor": stmt.excluded.adj_factor, "updated_at": now},
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_stk_limit(self, df: pd.DataFrame) -> int:
        now = datetime.utcnow()
        records = [
            {
                "ts_code": row.get("ts_code"),
                "trade_date": row.get("trade_date"),
                "up_limit": _clean_float(row.get("up_limit")),
                "down_limit": _clean_float(row.get("down_limit")),
                "updated_at": now,
            }
            for row in df.to_dict("records")
        ]
        with session_scope() as session:
            self._ensure_stock_placeholders(session, df)
            stmt = insert(StockLimitDaily)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_stock_limit_date",
                set_={"up_limit": stmt.excluded.up_limit, "down_limit": stmt.excluded.down_limit, "updated_at": now},
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _upsert_raw_records(self, endpoint: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        now = datetime.utcnow()
        records_by_key: dict[str, dict] = {}
        for row in df.to_dict("records"):
            payload = _json_payload(row)
            row_key = _raw_row_key(endpoint, payload)
            records_by_key[row_key] = {
                "endpoint": endpoint,
                "row_key": row_key,
                "ts_code": _string_or_none(payload.get("ts_code")),
                "trade_date": _date_string_or_none(payload.get("trade_date")),
                "ann_date": _date_string_or_none(payload.get("ann_date")),
                "f_ann_date": _date_string_or_none(payload.get("f_ann_date")),
                "end_date": _date_string_or_none(payload.get("end_date")),
                "period": _date_string_or_none(payload.get("period")) or _date_string_or_none(payload.get("end_date")),
                "payload": payload,
                "updated_at": now,
            }
        records = list(records_by_key.values())
        with session_scope() as session:
            stmt = insert(TushareRawRecord)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_tushare_raw_endpoint_row",
                set_={
                    "ts_code": stmt.excluded.ts_code,
                    "trade_date": stmt.excluded.trade_date,
                    "ann_date": stmt.excluded.ann_date,
                    "f_ann_date": stmt.excluded.f_ann_date,
                    "end_date": stmt.excluded.end_date,
                    "period": stmt.excluded.period,
                    "payload": stmt.excluded.payload,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            for batch in _batches(records):
                session.execute(stmt, batch)
        return len(records)

    def _index_codes_from_raw(self, markets: list[str] | None = None) -> list[str]:
        selected = markets or list(INDEX_BASIC_MARKETS)
        endpoints = [f"index_basic_{market.lower()}" for market in selected]
        with session_scope() as session:
            rows = (
                session.query(TushareRawRecord.payload)
                .filter(TushareRawRecord.endpoint.in_(endpoints))
                .all()
            )
        codes = set()
        for (payload,) in rows:
            if not isinstance(payload, dict):
                continue
            code = payload.get("ts_code") or payload.get("index_code")
            if code:
                codes.add(str(code))
        return sorted(codes)

    @staticmethod
    def _index_daily_date_counts(codes: list[str], start_date: str, end_date: str) -> dict[str, int]:
        if not codes:
            return {}
        counts: dict[str, int] = {}
        with session_scope() as session:
            for start in range(0, len(codes), 1000):
                chunk = codes[start : start + 1000]
                rows = (
                    session.query(IndexDaily.ts_code, func.count(func.distinct(IndexDaily.trade_date)))
                    .filter(
                        IndexDaily.ts_code.in_(chunk),
                        IndexDaily.trade_date >= start_date,
                        IndexDaily.trade_date <= end_date,
                    )
                    .group_by(IndexDaily.ts_code)
                    .all()
                )
                counts.update({row[0]: int(row[1]) for row in rows})
        return counts

    @staticmethod
    def _has_raw_records(endpoint: str, ts_code: str, start_date: str, end_date: str) -> bool:
        with session_scope() as session:
            count = (
                session.query(func.count())
                .select_from(TushareRawRecord)
                .filter(
                    TushareRawRecord.endpoint == endpoint,
                    TushareRawRecord.ts_code == ts_code,
                    (
                        ((TushareRawRecord.ann_date >= start_date) & (TushareRawRecord.ann_date <= end_date))
                        | ((TushareRawRecord.end_date >= start_date) & (TushareRawRecord.end_date <= end_date))
                        | ((TushareRawRecord.period >= start_date) & (TushareRawRecord.period <= end_date))
                    ),
                )
                .scalar()
                or 0
            )
        return int(count) > 0

    @staticmethod
    def _existing_raw_ts_codes(endpoints: list[str], start_date: str, end_date: str) -> dict[str, set[str]]:
        if not endpoints:
            return {}
        result = {endpoint: set() for endpoint in endpoints}
        with session_scope() as session:
            rows = (
                session.query(TushareRawRecord.endpoint, TushareRawRecord.ts_code)
                .filter(
                    TushareRawRecord.endpoint.in_(endpoints),
                    TushareRawRecord.ts_code.isnot(None),
                    (
                        ((TushareRawRecord.ann_date >= start_date) & (TushareRawRecord.ann_date <= end_date))
                        | ((TushareRawRecord.end_date >= start_date) & (TushareRawRecord.end_date <= end_date))
                        | ((TushareRawRecord.period >= start_date) & (TushareRawRecord.period <= end_date))
                    ),
                )
                .group_by(TushareRawRecord.endpoint, TushareRawRecord.ts_code)
                .all()
            )
        for endpoint, ts_code in rows:
            if ts_code:
                result.setdefault(endpoint, set()).add(ts_code)
        return result

    def _lookup_auxiliary_stock_metadata(self, ts_code: str) -> dict | None:
        try:
            namechange = self.client.query_endpoint("namechange", ts_code=ts_code)
        except Exception:
            namechange = pd.DataFrame()
        try:
            bak_basic = self.client.query_endpoint("bak_basic", ts_code=ts_code)
        except Exception:
            bak_basic = pd.DataFrame()

        name = area = industry = list_date = None
        if not namechange.empty and "name" in namechange.columns:
            frame = namechange.copy()
            if "start_date" in frame.columns:
                frame = frame.sort_values("start_date")
            name = _string_or_none(frame.iloc[-1].get("name"))
        if not bak_basic.empty:
            frame = bak_basic.copy()
            if "trade_date" in frame.columns:
                frame = frame.sort_values("trade_date")
            latest = frame.iloc[-1]
            name = name or _string_or_none(latest.get("name"))
            area = _string_or_none(latest.get("area"))
            industry = _string_or_none(latest.get("industry"))
            list_date = _date_string_or_none(latest.get("list_date"))
        if not name:
            return None
        return {
            "name": name,
            "area": area,
            "industry": industry,
            "market": "创业板" if ts_code.startswith(("300", "301")) else None,
            "list_date": list_date,
            "list_status": "D",
            "metadata_source": "namechange/bak_basic",
        }

    @staticmethod
    def _table_exists(table_name: str) -> bool:
        with session_scope() as session:
            result = session.execute(text("select to_regclass(:table_name)"), {"table_name": f"public.{table_name}"}).scalar()
            return result is not None

    @staticmethod
    def _legacy_table_frame(
        table_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if table_name not in set(LEGACY_RAW_FINANCIAL_TABLES.values()):
            raise DataSyncError(f"Unsupported legacy raw table: {table_name}")

        where_parts = []
        params: dict[str, str] = {}
        if start_date:
            params["start_date"] = start_date
            where_parts.append(
                "(ann_date >= :start_date OR f_ann_date >= :start_date OR end_date >= :start_date)"
            )
        if end_date:
            params["end_date"] = end_date
            where_parts.append(
                "(ann_date <= :end_date OR f_ann_date <= :end_date OR end_date <= :end_date)"
            )
        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        with session_scope() as session:
            columns = {
                row[0]
                for row in session.execute(
                    text(
                        "select column_name from information_schema.columns "
                        "where table_schema='public' and table_name=:table_name"
                    ),
                    {"table_name": table_name},
                )
            }
            cleaned_parts = []
            for part in where_parts:
                cleaned = part
                if "f_ann_date" not in columns:
                    cleaned = cleaned.replace(" OR f_ann_date >= :start_date", "")
                    cleaned = cleaned.replace(" OR f_ann_date <= :end_date", "")
                cleaned_parts.append(cleaned)
            where_sql = f" WHERE {' AND '.join(cleaned_parts)}" if cleaned_parts else ""
            rows = session.execute(text(f"select * from {table_name}{where_sql}"), params).mappings().all()
        return pd.DataFrame([dict(row) for row in rows])

    @staticmethod
    def _quarter_periods(start_date: str, end_date: str) -> list[str]:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        periods = []
        for year in range(start_year, end_year + 1):
            for suffix in ("0331", "0630", "0930", "1231"):
                period = f"{year}{suffix}"
                if start_date <= period <= end_date:
                    periods.append(period)
        return periods

    @staticmethod
    def _calendar_dates(start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()
        days = (end - start).days
        return [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days + 1)]

    @staticmethod
    def _ensure_stock_placeholders(session, df: pd.DataFrame) -> None:
        if df.empty or "ts_code" not in df.columns:
            return
        codes = sorted({str(code) for code in df["ts_code"].dropna().tolist()})
        if not codes:
            return
        existing = {
            row[0]
            for row in session.query(Stock.ts_code)
            .filter(Stock.ts_code.in_(codes))
            .all()
        }
        now = datetime.utcnow()
        for code in codes:
            if code in existing:
                continue
            symbol = code.split(".")[0]
            asset_class = _asset_class_for_ts_code(code)
            if asset_class == "fund_or_index_like":
                list_status = "N"
                industry = "NON_STOCK_ASSET"
                market = "NON_A_SHARE"
            else:
                list_status = "U"
                industry = None
                market = None
            stmt = insert(Stock).values(
                ts_code=code,
                symbol=symbol,
                name=f"UNKNOWN_{symbol}",
                area=None,
                industry=industry,
                market=market,
                list_date=None,
                list_status=list_status,
                is_hs=None,
                updated_at=now,
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=[Stock.ts_code])
            session.execute(stmt)

    def _record_run(
        self,
        task_type: str,
        status: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        affected_rows: int = 0,
        message: str | None = None,
    ) -> None:
        with session_scope() as session:
            session.add(
                DataSyncRun(
                    task_type=task_type,
                    status=status,
                    trade_date=trade_date,
                    start_date=start_date,
                    end_date=end_date,
                    affected_rows=affected_rows,
                    message=message,
                )
            )
