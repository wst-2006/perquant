"""Leakage-safe multi-horizon labels and partitioned datasets for v8 research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from ..db import get_engine


V8_HORIZONS = (1, 3, 5, 10, 20)
V8_THRESHOLDS = (0.05, 0.08, 0.12)
V8_LABEL_CONTRACT = "ml_v8_path_labels_v1"


class V8LabelError(RuntimeError):
    """Raised when v8 labels cannot be constructed from real market data."""


@dataclass(frozen=True)
class V8CostConfig:
    buy_fee_rate: float = 0.0003
    sell_fee_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0005


def realized_net_return(entry_open: float, exit_close: float, costs: V8CostConfig | None = None) -> float:
    cfg = costs or V8CostConfig()
    buy_cash = float(entry_open) * (1.0 + cfg.slippage) * (1.0 + cfg.buy_fee_rate)
    sell_cash = float(exit_close) * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee_rate - cfg.stamp_tax_rate)
    return sell_cash / buy_cash - 1.0


def entry_is_tradable(open_price: float | None, up_limit: float | None) -> bool:
    return bool(
        open_price is not None
        and up_limit is not None
        and np.isfinite(open_price)
        and np.isfinite(up_limit)
        and open_price > 0
        and open_price < up_limit - 1e-8
    )


def exit_is_tradable(close_price: float | None, down_limit: float | None) -> bool:
    return bool(
        close_price is not None
        and down_limit is not None
        and np.isfinite(close_price)
        and np.isfinite(down_limit)
        and close_price > 0
        and close_price > down_limit + 1e-8
    )


def compute_path_labels(
    entry: dict[str, Any] | None,
    future_sell_days: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...] = V8_HORIZONS,
    thresholds: tuple[float, ...] = V8_THRESHOLDS,
    costs: V8CostConfig | None = None,
) -> dict[str, Any]:
    """Reference implementation used by audits and boundary tests.

    ``future_sell_days[0]`` is the earliest legal T+1 sell date after buying at
    ``entry``. Maximum-close opportunity labels only consider sellable closes.
    Intraday highs are diagnostic and never promotion labels.
    """
    cfg = costs or V8CostConfig()
    entry = entry or {}
    entry_open = _float(entry.get("open"))
    entry_up_limit = _float(entry.get("up_limit"))
    tradable = entry_is_tradable(entry_open, entry_up_limit)
    output: dict[str, Any] = {
        "label_entry_date": _date(entry.get("trade_date")),
        "label_entry_tradable": tradable,
        "label_entry_reason": "tradable" if tradable else _entry_reason(entry_open, entry_up_limit),
    }
    for horizon in horizons:
        prefix = f"label_{horizon}d"
        path = future_sell_days[:horizon]
        fixed = path[horizon - 1] if len(path) >= horizon else None
        fixed_close = _float((fixed or {}).get("close"))
        fixed_down = _float((fixed or {}).get("down_limit"))
        fixed_tradable = tradable and exit_is_tradable(fixed_close, fixed_down)
        output[f"{prefix}_end_date"] = _date((fixed or {}).get("trade_date"))
        output[f"{prefix}_fixed_tradable"] = fixed_tradable
        output[f"{prefix}_net_return"] = realized_net_return(entry_open, fixed_close, cfg) if fixed_tradable else None

        sellable_returns: list[float] = []
        close_marks: list[float] = []
        high_marks: list[float] = []
        for day in path:
            close = _float(day.get("close"))
            high = _float(day.get("high"))
            down_limit = _float(day.get("down_limit"))
            if tradable and close is not None and close > 0:
                mark = realized_net_return(entry_open, close, cfg)
                close_marks.append(mark)
                if exit_is_tradable(close, down_limit):
                    sellable_returns.append(mark)
            if tradable and high is not None and high > 0:
                high_marks.append(realized_net_return(entry_open, high, cfg))

        mfe = max(sellable_returns) if sellable_returns else None
        mae = min(close_marks) if close_marks else None
        output[f"{prefix}_mfe_close"] = mfe
        output[f"{prefix}_mae_close"] = mae
        output[f"{prefix}_mfe_high_diagnostic"] = max(high_marks) if high_marks else None
        for threshold in thresholds:
            key = _threshold_key(threshold)
            output[f"{prefix}_hit_{key}"] = bool(mfe is not None and mfe >= threshold) if tradable else None
            first = None
            if tradable:
                for index, day in enumerate(path, start=1):
                    close = _float(day.get("close"))
                    down_limit = _float(day.get("down_limit"))
                    if close is None or not exit_is_tradable(close, down_limit):
                        continue
                    if realized_net_return(entry_open, close, cfg) >= threshold:
                        first = index
                        break
            output[f"{prefix}_time_to_{key}"] = first
    return output


class V8MarketPanel:
    """Compact real quote/limit panel for vectorized path-label construction."""

    FIELDS = ("open", "close", "high", "low", "up_limit", "down_limit")

    def __init__(self, dates: list[str], codes: list[str], industries: dict[str, str | None], arrays: dict[str, np.ndarray]) -> None:
        self.dates = dates
        self.codes = codes
        self.industries = industries
        self.arrays = arrays
        self.date_to_index = {value: index for index, value in enumerate(dates)}
        self.code_to_index = {value: index for index, value in enumerate(codes)}

    @classmethod
    def from_database(cls, start_date: str, end_date: str | None = None, chunksize: int = 200_000) -> "V8MarketPanel":
        engine = get_engine()
        connection = engine.raw_connection()
        try:
            with connection.cursor() as cursor:
                date_sql = "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date >= %s"
                params: list[str] = [start_date]
                if end_date:
                    date_sql += " AND trade_date <= %s"
                    params.append(end_date)
                date_sql += " ORDER BY trade_date"
                cursor.execute(date_sql, params)
                dates = [str(row[0]) for row in cursor.fetchall()]
                cursor.execute("SELECT ts_code, industry FROM stocks")
                stock_rows = cursor.fetchall()
            codes = sorted({str(row[0]) for row in stock_rows})
            industries = {str(row[0]): row[1] for row in stock_rows}
            if not dates or not codes:
                raise V8LabelError("Database has no quote dates or stock codes for v8 labels.")
            arrays = {name: np.full((len(codes), len(dates)), np.nan, dtype=np.float32) for name in cls.FIELDS}
            panel = cls(dates, codes, industries, arrays)
            quote_sql = (
                "SELECT ts_code, trade_date, open, close, high, low FROM daily_quotes "
                "WHERE trade_date >= %s AND trade_date <= %s"
            )
            for chunk in _cursor_chunks(connection, quote_sql, [dates[0], dates[-1]], ("ts_code", "trade_date", "open", "close", "high", "low"), chunksize):
                panel._fill_chunk(chunk, ("open", "close", "high", "low"))
            limit_sql = (
                "SELECT ts_code, trade_date, up_limit, down_limit FROM stock_limit_daily "
                "WHERE trade_date >= %s AND trade_date <= %s"
            )
            for chunk in _cursor_chunks(connection, limit_sql, [dates[0], dates[-1]], ("ts_code", "trade_date", "up_limit", "down_limit"), chunksize):
                panel._fill_chunk(chunk, ("up_limit", "down_limit"))
        finally:
            connection.close()
        return panel

    @classmethod
    def from_frames(cls, quotes: pd.DataFrame, limits: pd.DataFrame, stocks: pd.DataFrame) -> "V8MarketPanel":
        dates = sorted(set(quotes["trade_date"].astype(str)) | set(limits["trade_date"].astype(str)))
        codes = sorted(stocks["ts_code"].astype(str).unique().tolist())
        industries = dict(zip(stocks["ts_code"].astype(str), stocks.get("industry", pd.Series([None] * len(stocks))), strict=False))
        arrays = {name: np.full((len(codes), len(dates)), np.nan, dtype=np.float32) for name in cls.FIELDS}
        panel = cls(dates, codes, industries, arrays)
        panel._fill_chunk(quotes, ("open", "close", "high", "low"))
        panel._fill_chunk(limits, ("up_limit", "down_limit"))
        return panel

    def _fill_chunk(self, chunk: pd.DataFrame, fields: tuple[str, ...]) -> None:
        code_idx = chunk["ts_code"].astype(str).map(self.code_to_index)
        date_idx = chunk["trade_date"].astype(str).map(self.date_to_index)
        usable = code_idx.notna() & date_idx.notna()
        rows = code_idx[usable].astype(np.int32).to_numpy()
        cols = date_idx[usable].astype(np.int32).to_numpy()
        for field in fields:
            values = pd.to_numeric(chunk.loc[usable, field], errors="coerce").astype(np.float32).to_numpy()
            self.arrays[field][rows, cols] = values

    def labels_for_rows(
        self,
        rows: pd.DataFrame,
        *,
        horizons: tuple[int, ...] = V8_HORIZONS,
        thresholds: tuple[float, ...] = V8_THRESHOLDS,
        costs: V8CostConfig | None = None,
    ) -> pd.DataFrame:
        cfg = costs or V8CostConfig()
        code_idx = rows["ts_code"].astype(str).map(self.code_to_index)
        date_idx = rows["trade_date"].map(_date).map(self.date_to_index)
        valid_keys = code_idx.notna() & date_idx.notna()
        ci = code_idx.fillna(0).astype(np.int32).to_numpy()
        di = date_idx.fillna(0).astype(np.int32).to_numpy()
        width = len(self.dates)

        entry_pos = di + 1
        entry_in_range = valid_keys.to_numpy() & (entry_pos < width)
        entry_open = self._take("open", ci, entry_pos, entry_in_range)
        entry_up = self._take("up_limit", ci, entry_pos, entry_in_range)
        entry_tradable = entry_in_range & np.isfinite(entry_open) & np.isfinite(entry_up) & (entry_open > 0) & (entry_open < entry_up - 1e-8)
        output: dict[str, Any] = {
            "label_entry_date": self._date_values(entry_pos, entry_in_range),
            "label_entry_tradable": entry_tradable,
            "label_entry_reason": np.where(
                entry_tradable,
                "tradable",
                np.where(~entry_in_range, "missing_market_date", np.where(~np.isfinite(entry_open), "missing_open", np.where(~np.isfinite(entry_up), "missing_up_limit", "limit_up_open"))),
            ),
        }
        for horizon in horizons:
            prefix = f"label_{horizon}d"
            offsets = np.arange(2, horizon + 2, dtype=np.int32)
            positions = di[:, None] + offsets[None, :]
            in_range = valid_keys.to_numpy()[:, None] & (positions < width)
            close_path = self._take_matrix("close", ci, positions, in_range)
            high_path = self._take_matrix("high", ci, positions, in_range)
            down_path = self._take_matrix("down_limit", ci, positions, in_range)
            sellable = entry_tradable[:, None] & np.isfinite(close_path) & np.isfinite(down_path) & (close_path > 0) & (close_path > down_path + 1e-8)
            close_available = entry_tradable[:, None] & np.isfinite(close_path) & (close_path > 0)
            high_available = entry_tradable[:, None] & np.isfinite(high_path) & (high_path > 0)
            close_returns = self._return_matrix(entry_open, close_path, cfg)
            high_returns = self._return_matrix(entry_open, high_path, cfg)

            fixed_valid = sellable[:, -1]
            output[f"{prefix}_end_date"] = self._date_values(positions[:, -1], in_range[:, -1])
            output[f"{prefix}_fixed_tradable"] = fixed_valid
            output[f"{prefix}_net_return"] = np.where(fixed_valid, close_returns[:, -1], np.nan).astype(np.float32)
            mfe = _row_extreme(np.where(sellable, close_returns, np.nan), "max")
            output[f"{prefix}_mfe_close"] = mfe
            output[f"{prefix}_mae_close"] = _row_extreme(np.where(close_available, close_returns, np.nan), "min")
            output[f"{prefix}_mfe_high_diagnostic"] = _row_extreme(np.where(high_available, high_returns, np.nan), "max")
            for threshold in thresholds:
                key = _threshold_key(threshold)
                reached = sellable & (close_returns >= threshold)
                hit = reached.any(axis=1)
                output[f"{prefix}_hit_{key}"] = pd.array(np.where(entry_tradable, hit, None), dtype="boolean")
                first = np.where(hit, reached.argmax(axis=1) + 1, 0).astype(np.int16)
                output[f"{prefix}_time_to_{key}"] = pd.array(np.where(hit, first, None), dtype="Int16")
        return pd.DataFrame(output, index=rows.index)

    def industry_for_codes(self, codes: pd.Series) -> pd.Series:
        return codes.astype(str).map(self.industries)

    def _take(self, field: str, rows: np.ndarray, cols: np.ndarray, valid: np.ndarray) -> np.ndarray:
        safe_cols = np.clip(cols, 0, len(self.dates) - 1)
        return np.where(valid, self.arrays[field][rows, safe_cols].astype(np.float64), np.nan)

    def _take_matrix(self, field: str, rows: np.ndarray, cols: np.ndarray, valid: np.ndarray) -> np.ndarray:
        safe_cols = np.clip(cols, 0, len(self.dates) - 1)
        return np.where(valid, self.arrays[field][rows[:, None], safe_cols].astype(np.float64), np.nan)

    def _date_values(self, positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
        dates = np.asarray(self.dates, dtype=object)
        return np.where(valid, dates[np.clip(positions, 0, len(self.dates) - 1)], None)

    @staticmethod
    def _return_matrix(entry_open: np.ndarray, exit_price: np.ndarray, cfg: V8CostConfig) -> np.ndarray:
        buy = entry_open[:, None] * (1.0 + cfg.slippage) * (1.0 + cfg.buy_fee_rate)
        sell = exit_price * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee_rate - cfg.stamp_tax_rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            return sell / buy - 1.0


@dataclass
class V8DatasetBuilder:
    panel: V8MarketPanel
    costs: V8CostConfig = V8CostConfig()

    def build(self, source_path: str | Path, output_dir: str | Path, *, chunksize: int = 200_000, feature_set: str = "factor_lab_v8") -> dict[str, Any]:
        source = Path(source_path)
        target = Path(output_dir)
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() and any(target.iterdir()):
            raise V8LabelError(f"Output directory must be empty or absent: {target}")
        target.mkdir(parents=True, exist_ok=True)

        columns = pd.read_csv(source, nrows=0).columns.tolist()
        keep = [col for col in columns if not col.startswith("label_")]
        if not {"trade_date", "ts_code"}.issubset(keep):
            raise V8LabelError("Source frame must include trade_date and ts_code.")
        rows_written = 0
        dates_written: list[str] = []
        label_counts: dict[str, int] = {}
        for chunk in _complete_date_chunks(source, keep, chunksize):
            chunk["trade_date"] = chunk["trade_date"].map(_date)
            labels = self.panel.labels_for_rows(chunk, costs=self.costs)
            chunk = pd.concat([chunk.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
            chunk["industry"] = self.panel.industry_for_codes(chunk["ts_code"])
            float_columns = [name for name in chunk.columns if name.startswith("feature_") or (name.startswith("label_") and chunk[name].dtype.kind == "f")]
            for col in float_columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)
            for trade_date, daily in chunk.groupby("trade_date", sort=False):
                partition = target / f"trade_date={trade_date}"
                partition.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(partition / "part-000.parquet", index=False, engine="pyarrow", compression="zstd")
                dates_written.append(str(trade_date))
                rows_written += len(daily)
            for col in [name for name in labels.columns if name.endswith("_net_return") or "_mfe_close" in name or "_mae_close" in name]:
                label_counts[col] = label_counts.get(col, 0) + int(pd.to_numeric(labels[col], errors="coerce").notna().sum())

        manifest = {
            "task": "build_v8_partitioned_dataset",
            "label_contract": V8_LABEL_CONTRACT,
            "feature_set": feature_set,
            "source_path": str(source),
            "source_sha256": _file_sha256(source),
            "output_dir": str(target),
            "rows": int(rows_written),
            "date_count": len(set(dates_written)),
            "date_range": {"start": min(dates_written) if dates_written else None, "end": max(dates_written) if dates_written else None},
            "horizons": list(V8_HORIZONS),
            "thresholds": list(V8_THRESHOLDS),
            "costs": asdict(self.costs),
            "label_non_null_counts": label_counts,
            "primary_label_policy": "next-open entry; earliest sell is the following trading day; promotion uses sellable future closes, never intraday highs",
            "data_policy": "real Tushare-backed PostgreSQL quotes and stock-specific limits only; no simulated values",
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}


def _complete_date_chunks(source: Path, columns: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    carry = pd.DataFrame()
    previous = ""
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        chunk["trade_date"] = chunk["trade_date"].map(_date)
        dates = chunk["trade_date"]
        if not dates.is_monotonic_increasing or (previous and dates.iloc[0] < previous):
            raise V8LabelError("Source training frame must be sorted by trade_date.")
        previous = dates.iloc[-1]
        combined = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        last = combined["trade_date"].iloc[-1]
        complete = combined[combined["trade_date"] != last]
        carry = combined[combined["trade_date"] == last].copy()
        if not complete.empty:
            yield complete
    if not carry.empty:
        yield carry


def _cursor_chunks(connection: Any, sql: str, params: list[Any], columns: tuple[str, ...], chunksize: int) -> Iterator[pd.DataFrame]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        while rows := cursor.fetchmany(chunksize):
            yield pd.DataFrame.from_records(rows, columns=columns)


def _row_extreme(values: np.ndarray, operation: str) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf if operation == "max" else np.inf)
    result = filled.max(axis=1) if operation == "max" else filled.min(axis=1)
    return np.where(finite.any(axis=1), result, np.nan).astype(np.float32)


def _entry_reason(open_price: float | None, up_limit: float | None) -> str:
    if open_price is None:
        return "missing_open"
    if up_limit is None:
        return "missing_up_limit"
    if open_price <= 0:
        return "invalid_open"
    return "limit_up_open"


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "")[:8]


def _threshold_key(value: float) -> str:
    return f"{int(round(value * 100))}pct"


def _file_sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()
