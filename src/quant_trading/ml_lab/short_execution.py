"""Auditable daily-bar proxy for the user's short-horizon execution policy.

The target entry time is 09:35 on D+1, but this project does not currently
have audited minute bars. Historical labels therefore use the D+1 daily open
plus explicit slippage and never claim to represent an observed 09:35 fill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHORT_EXECUTION_CONTRACT = "short_execution_daily_proxy_v1"


@dataclass(frozen=True)
class ShortExecutionCostConfig:
    buy_commission_rate: float = 0.0003
    sell_commission_rate: float = 0.0003
    sell_stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    minimum_commission_mode: str = "notional_unknown_not_applied"


@dataclass(frozen=True)
class ShortExecutionPolicy:
    entry_time_target: str = "09:35"
    entry_price_source: str = "daily_open_proxy"
    take_profit_rate: float = 0.01
    stop_loss_rate: float = 0.05
    first_legal_sell_offset: int = 1
    terminal_sell_offset: int = 4
    same_bar_rule: str = "stop_first"
    defer_blocked_terminal_exit: bool = True


def compute_short_execution_labels_for_panel(
    panel: Any,
    rows: pd.DataFrame,
    *,
    policy: ShortExecutionPolicy | None = None,
    costs: ShortExecutionCostConfig | None = None,
    max_exit_scan_days: int = 10,
) -> pd.DataFrame:
    """Vectorized counterpart of :func:`compute_short_execution_label`.

    The function consumes the audited ``V8MarketPanel`` interface without
    importing it, keeping the pure execution contract free of circular imports.
    """
    rule = policy or ShortExecutionPolicy()
    fee = costs or ShortExecutionCostConfig()
    if max_exit_scan_days < rule.terminal_sell_offset:
        raise ValueError("max_exit_scan_days must include the terminal sell day")
    if rule.same_bar_rule != "stop_first":
        raise ValueError("Only the conservative stop_first daily-bar rule is supported")

    count = len(rows)
    code_idx = rows["ts_code"].astype(str).map(panel.code_to_index)
    date_idx = rows["trade_date"].astype(str).str.replace("-", "", regex=False).map(panel.date_to_index)
    valid_keys = (code_idx.notna() & date_idx.notna()).to_numpy()
    ci = code_idx.fillna(0).astype(np.int32).to_numpy()
    di = date_idx.fillna(0).astype(np.int32).to_numpy()
    width = len(panel.dates)

    def take(field: str, positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
        safe = np.clip(positions, 0, max(width - 1, 0))
        if positions.ndim == 1:
            values = panel.arrays[field][ci, safe].astype(float)
        else:
            values = panel.arrays[field][ci[:, None], safe].astype(float)
        return np.where(valid, values, np.nan)

    entry_pos = di + 1
    entry_valid = valid_keys & (entry_pos < width)
    entry_open = take("open", entry_pos, entry_valid)
    entry_close = take("close", entry_pos, entry_valid)
    entry_low = take("low", entry_pos, entry_valid)
    entry_up = take("up_limit", entry_pos, entry_valid)
    entry_tradable = entry_valid & np.isfinite(entry_open) & np.isfinite(entry_up) & (entry_open > 0) & (entry_open < entry_up - 1e-8)
    entry_execution = entry_open * (1.0 + fee.slippage_rate)
    buy_cash = entry_execution * (1.0 + fee.buy_commission_rate)
    tp_price = entry_execution * (1.0 + rule.take_profit_rate)
    sl_price = entry_execution * (1.0 - rule.stop_loss_rate)

    offsets = np.arange(2, max_exit_scan_days + 2, dtype=np.int32)
    positions = di[:, None] + offsets[None, :]
    in_range = valid_keys[:, None] & (positions < width)
    opens = take("open", positions, in_range)
    closes = take("close", positions, in_range)
    highs = take("high", positions, in_range)
    lows = take("low", positions, in_range)
    downs = take("down_limit", positions, in_range)
    available = in_range & np.isfinite(opens) & np.isfinite(closes) & np.isfinite(highs) & np.isfinite(lows) & np.isfinite(downs)

    settled = ~entry_tradable
    status = np.where(entry_tradable, "pending", "entry_untradable").astype(object)
    exit_index = np.full(count, -1, dtype=np.int16)
    exit_reason = np.full(count, None, dtype=object)
    exit_source = np.full(count, None, dtype=object)
    exit_reference = np.full(count, np.nan, dtype=float)
    realized = np.full(count, np.nan, dtype=float)
    tp_hit = np.zeros(count, dtype=bool)
    sl_hit = np.zeros(count, dtype=bool)
    tp_before_sl = np.full(count, np.nan, dtype=float)
    gap_through = np.zeros(count, dtype=bool)
    ambiguous = np.zeros(count, dtype=bool)
    blocked_count = np.zeros(count, dtype=np.int16)

    sell_multiplier = (1.0 - fee.slippage_rate) * (1.0 - fee.sell_commission_rate - fee.sell_stamp_tax_rate)

    def finish(mask: np.ndarray, k: int, reference: np.ndarray, reason: str, source: str, *, tp: bool, sl: bool, gap: np.ndarray | None = None) -> None:
        if not mask.any():
            return
        settled[mask] = True
        status[mask] = "settled"
        exit_index[mask] = k
        exit_reason[mask] = reason
        exit_source[mask] = source
        exit_reference[mask] = reference[mask]
        realized[mask] = reference[mask] * sell_multiplier / buy_cash[mask] - 1.0
        tp_hit[mask] |= tp
        sl_hit[mask] |= sl
        if tp:
            tp_before_sl[mask] = 1.0
        elif sl:
            tp_before_sl[mask] = 0.0
        if gap is not None:
            gap_through[mask] |= gap[mask]

    for k in range(max_exit_scan_days):
        active = entry_tradable & ~settled & in_range[:, k]
        if not active.any():
            continue
        terminal = k + 1 >= rule.terminal_sell_offset
        gap_sl = available[:, k] & (opens[:, k] <= sl_price)
        gap_tp = available[:, k] & (opens[:, k] >= tp_price)
        hit_sl = available[:, k] & (lows[:, k] <= sl_price)
        hit_tp = available[:, k] & (highs[:, k] >= tp_price)
        open_sellable = available[:, k] & (opens[:, k] > downs[:, k] + 1e-8)
        blocked = active & (~available[:, k] | (~open_sellable & (gap_sl | hit_sl | terminal))) & (terminal | gap_sl | hit_sl)
        blocked_count[blocked] += 1
        sl_hit[blocked & (gap_sl | hit_sl)] = True
        gap_through[blocked & gap_sl] = True

        eligible = active & open_sellable
        mask = eligible & gap_sl
        finish(mask, k, opens[:, k], "gap_stop", "daily_open_proxy", tp=False, sl=True, gap=opens[:, k] < sl_price - 1e-8)
        eligible &= ~settled
        mask = eligible & gap_tp
        finish(mask, k, opens[:, k], "gap_take_profit", "daily_open_proxy", tp=True, sl=False)
        eligible &= ~settled
        mask = eligible & hit_tp & hit_sl
        ambiguous[mask] = True
        finish(mask, k, sl_price, "same_bar_stop_first", "daily_threshold_proxy", tp=False, sl=True)
        eligible &= ~settled
        mask = eligible & hit_sl
        finish(mask, k, sl_price, "stop_loss", "daily_threshold_proxy", tp=False, sl=True)
        eligible &= ~settled
        mask = eligible & hit_tp
        finish(mask, k, tp_price, "take_profit", "daily_threshold_proxy", tp=True, sl=False)
        eligible &= ~settled
        if terminal:
            close_sellable = available[:, k] & (closes[:, k] > downs[:, k] + 1e-8)
            mask = eligible & close_sellable
            reason = "terminal_close" if k + 1 == rule.terminal_sell_offset else "terminal_close_deferred"
            finish(mask, k, closes[:, k], reason, "daily_close", tp=False, sl=False)
            blocked_close = eligible & ~settled & ~close_sellable
            blocked_count[blocked_close] += 1

    terminal_pos = di + rule.terminal_sell_offset + 1
    terminal_reached = valid_keys & (terminal_pos < width)
    pending_blocked = entry_tradable & ~settled & terminal_reached & (blocked_count > 0)
    status[pending_blocked] = "pending_blocked_exit"

    with np.errstate(invalid="ignore", divide="ignore"):
        entry_low_mark = entry_low * sell_multiplier / buy_cash - 1.0
        low_marks = lows * sell_multiplier / buy_cash[:, None] - 1.0
    path_mask = np.ones_like(low_marks, dtype=bool)
    for k in range(max_exit_scan_days):
        path_mask[:, k] &= (exit_index < 0) | (k <= exit_index)
    adverse = np.concatenate([entry_low_mark[:, None], np.where(path_mask, low_marks, np.nan)], axis=1)
    adverse[~np.isfinite(adverse)] = np.nan
    has_adverse = np.isfinite(adverse).any(axis=1) & entry_tradable
    max_adverse = np.full(count, np.nan, dtype=float)
    max_adverse[has_adverse] = np.nanmin(adverse[has_adverse], axis=1)

    locked_return = entry_close * sell_multiplier / buy_cash - 1.0
    locked_return[~entry_tradable | ~np.isfinite(entry_close)] = np.nan
    exit_date = np.full(count, None, dtype=object)
    for k in range(max_exit_scan_days):
        mask = exit_index == k
        if mask.any():
            exit_date[mask] = np.asarray(panel.dates, dtype=object)[positions[mask, k]]
    entry_dates = np.full(count, None, dtype=object)
    entry_dates[entry_valid] = np.asarray(panel.dates, dtype=object)[entry_pos[entry_valid]]
    terminal_dates = np.full(count, None, dtype=object)
    terminal_dates[terminal_reached] = np.asarray(panel.dates, dtype=object)[terminal_pos[terminal_reached]]
    first_sell_dates = np.full(count, None, dtype=object)
    first_valid = valid_keys & (di + 2 < width)
    first_sell_dates[first_valid] = np.asarray(panel.dates, dtype=object)[(di + 2)[first_valid]]
    entry_reason = np.where(
        entry_tradable,
        "tradable",
        np.where(~entry_valid, "missing_market_date", np.where(~np.isfinite(entry_open), "missing_open", np.where(~np.isfinite(entry_up), "missing_up_limit", "limit_up_open"))),
    )

    return pd.DataFrame(
        {
            "short_contract": SHORT_EXECUTION_CONTRACT,
            "short_status": status,
            "short_settled": settled,
            "short_entry_date": entry_dates,
            "short_entry_tradable": entry_tradable,
            "short_entry_reason": entry_reason,
            "short_first_legal_sell_date": first_sell_dates,
            "short_terminal_target_date": terminal_dates,
            "short_locked_day_return": locked_return.astype(np.float32),
            "short_realized_net_return": realized.astype(np.float32),
            "short_positive": np.where(np.isfinite(realized), realized > 0, None),
            "short_exit_date": exit_date,
            "short_exit_reason": exit_reason,
            "short_exit_price_source": exit_source,
            "short_exit_reference_price": exit_reference.astype(np.float32),
            "short_take_profit_hit": tp_hit,
            "short_stop_loss_hit": sl_hit,
            "short_tp_before_sl": np.where(np.isfinite(tp_before_sl), tp_before_sl.astype(bool), None),
            "short_gap_through_stop": gap_through,
            "short_ambiguous_daily_bar": ambiguous,
            "short_blocked_exit_days": blocked_count,
            "short_exit_untradable": pending_blocked,
            "short_max_adverse_path": max_adverse.astype(np.float32),
        }
    )


@dataclass
class ShortExecutionLabelAugmenter:
    panel: Any
    policy: ShortExecutionPolicy = ShortExecutionPolicy()
    costs: ShortExecutionCostConfig = ShortExecutionCostConfig()
    max_exit_scan_days: int = 10

    def build(self, source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        source = Path(source_dir)
        target = Path(output_dir)
        partitions = sorted(source.glob("trade_date=*/part-000.parquet"), key=lambda path: path.parent.name)
        if not partitions:
            raise FileNotFoundError(f"No source partitions found in {source}")
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Output directory must be empty or absent: {target}")
        target.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        rows_written = 0
        dates: list[str] = []
        for path in partitions:
            keys = pd.read_parquet(path, columns=["trade_date", "ts_code"])
            labels = compute_short_execution_labels_for_panel(
                self.panel,
                keys,
                policy=self.policy,
                costs=self.costs,
                max_exit_scan_days=self.max_exit_scan_days,
            )
            output = pd.concat([keys.reset_index(drop=True), labels], axis=1)
            partition = target / path.parent.name
            partition.mkdir(parents=True, exist_ok=True)
            output.to_parquet(partition / "part-000.parquet", index=False, engine="pyarrow", compression="zstd")
            rows_written += len(output)
            dates.append(path.parent.name.split("=", 1)[1])
            for column in ("short_settled", "short_positive", "short_take_profit_hit", "short_stop_loss_hit", "short_exit_untradable"):
                counts[column] = counts.get(column, 0) + int(pd.Series(output[column]).fillna(False).astype(bool).sum())
        source_manifest = source / "manifest.json"
        manifest = {
            "task": "build_short_execution_label_sidecar",
            "contract": SHORT_EXECUTION_CONTRACT,
            "source_dataset_dir": str(source),
            "source_manifest_sha256": _file_sha256(source_manifest) if source_manifest.exists() else None,
            "output_dir": str(target),
            "rows": rows_written,
            "date_count": len(dates),
            "date_range": {"start": min(dates), "end": max(dates)},
            "policy": asdict(self.policy),
            "costs": asdict(self.costs),
            "max_exit_scan_days": self.max_exit_scan_days,
            "counts": counts,
            "data_policy": "real daily quotes and stock-specific limits; daily open is an explicit 09:35 proxy",
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_short_execution_label(
    entry: dict[str, Any] | None,
    future_sell_days: list[dict[str, Any]],
    *,
    policy: ShortExecutionPolicy | None = None,
    costs: ShortExecutionCostConfig | None = None,
) -> dict[str, Any]:
    """Evaluate D+1 entry and D+2 onward exits from audited daily bars.

    ``entry`` is D+1 and ``future_sell_days[0]`` is D+2, the first legal sell
    day. When a daily bar crosses both thresholds, intraday order is unknown,
    so the conservative stop-first rule is used.
    """
    rule = policy or ShortExecutionPolicy()
    fee = costs or ShortExecutionCostConfig()
    if rule.same_bar_rule != "stop_first":
        raise ValueError("Only the conservative stop_first daily-bar rule is supported")
    if rule.first_legal_sell_offset != 1 or rule.terminal_sell_offset < 1:
        raise ValueError("Invalid T+1 short-execution offsets")

    entry = entry or {}
    entry_open = _price(entry.get("open"))
    entry_close = _price(entry.get("close"))
    entry_low = _price(entry.get("low"))
    entry_up_limit = _price(entry.get("up_limit"))
    entry_tradable = _entry_tradable(entry_open, entry_up_limit)
    entry_execution = entry_open * (1.0 + fee.slippage_rate) if entry_tradable else None
    buy_cash = entry_execution * (1.0 + fee.buy_commission_rate) if entry_execution else None

    result: dict[str, Any] = {
        "contract": SHORT_EXECUTION_CONTRACT,
        "status": "pending",
        "settled": False,
        "entry_date": _date(entry.get("trade_date")),
        "entry_time_target": rule.entry_time_target,
        "entry_price_source": rule.entry_price_source,
        "entry_reference_price": entry_open,
        "entry_execution_price": entry_execution,
        "entry_tradable": entry_tradable,
        "entry_reason": "tradable" if entry_tradable else _entry_reason(entry_open, entry_up_limit),
        "proxy_warning": "09:35 minute price unavailable; D+1 daily open plus slippage is an explicit historical proxy",
        "first_legal_sell_date": _date(future_sell_days[0].get("trade_date")) if future_sell_days else None,
        "terminal_target_date": _date(future_sell_days[rule.terminal_sell_offset - 1].get("trade_date"))
        if len(future_sell_days) >= rule.terminal_sell_offset
        else None,
        "locked_day_return": _net_mark_return(buy_cash, entry_close, fee),
        "realized_net_return": None,
        "exit_date": None,
        "exit_reason": None,
        "exit_price_source": None,
        "exit_reference_price": None,
        "exit_execution_price": None,
        "exit_tradable": None,
        "take_profit_hit": False,
        "stop_loss_hit": False,
        "tp_before_sl": None,
        "gap_through_stop": False,
        "ambiguous_daily_bar": False,
        "blocked_exit_dates": [],
        "max_adverse_path": None,
        "available_sell_days": len(future_sell_days),
        "policy": asdict(rule),
        "costs": asdict(fee),
    }
    if not entry_tradable or buy_cash is None or entry_execution is None:
        result.update(status="entry_untradable", settled=True, exit_tradable=False)
        return result

    adverse_marks = [
        value
        for value in (_net_mark_return(buy_cash, entry_low, fee),)
        if value is not None
    ]
    take_profit_price = entry_execution * (1.0 + rule.take_profit_rate)
    stop_loss_price = entry_execution * (1.0 - rule.stop_loss_rate)
    terminal_reached = len(future_sell_days) >= rule.terminal_sell_offset

    for index, day in enumerate(future_sell_days, start=1):
        open_price = _price(day.get("open"))
        close_price = _price(day.get("close"))
        high_price = _price(day.get("high"))
        low_price = _price(day.get("low"))
        down_limit = _price(day.get("down_limit"))
        trade_date = _date(day.get("trade_date"))
        low_mark = _net_mark_return(buy_cash, low_price, fee)
        if low_mark is not None:
            adverse_marks.append(low_mark)

        day_available = all(value is not None for value in (open_price, close_price, high_price, low_price, down_limit))
        open_sellable = bool(day_available and open_price > down_limit + 1e-8)
        terminal_day_or_later = index >= rule.terminal_sell_offset

        if not day_available:
            if terminal_day_or_later:
                result["blocked_exit_dates"].append({"trade_date": trade_date, "reason": "missing_or_suspended_bar"})
            continue

        gap_tp = open_price >= take_profit_price
        gap_sl = open_price <= stop_loss_price
        hit_tp = high_price >= take_profit_price
        hit_sl = low_price <= stop_loss_price

        if not open_sellable:
            if gap_sl or hit_sl or terminal_day_or_later:
                result["blocked_exit_dates"].append({"trade_date": trade_date, "reason": "down_limit_open_or_untradable"})
                result["stop_loss_hit"] = bool(result["stop_loss_hit"] or gap_sl or hit_sl)
                result["gap_through_stop"] = bool(result["gap_through_stop"] or gap_sl)
            continue

        if gap_sl:
            return _finish(
                result,
                buy_cash,
                open_price,
                trade_date,
                "gap_stop",
                "daily_open_proxy",
                fee,
                adverse_marks,
                take_profit=False,
                stop_loss=True,
                gap_through=open_price < stop_loss_price - 1e-8,
            )
        if gap_tp:
            return _finish(
                result,
                buy_cash,
                open_price,
                trade_date,
                "gap_take_profit",
                "daily_open_proxy",
                fee,
                adverse_marks,
                take_profit=True,
                stop_loss=False,
            )
        if hit_tp and hit_sl:
            result["ambiguous_daily_bar"] = True
            return _finish(
                result,
                buy_cash,
                stop_loss_price,
                trade_date,
                "same_bar_stop_first",
                "daily_threshold_proxy",
                fee,
                adverse_marks,
                take_profit=False,
                stop_loss=True,
            )
        if hit_sl:
            return _finish(
                result,
                buy_cash,
                stop_loss_price,
                trade_date,
                "stop_loss",
                "daily_threshold_proxy",
                fee,
                adverse_marks,
                take_profit=False,
                stop_loss=True,
            )
        if hit_tp:
            return _finish(
                result,
                buy_cash,
                take_profit_price,
                trade_date,
                "take_profit",
                "daily_threshold_proxy",
                fee,
                adverse_marks,
                take_profit=True,
                stop_loss=False,
            )
        if terminal_day_or_later and _close_sellable(close_price, down_limit):
            reason = "terminal_close" if index == rule.terminal_sell_offset else "terminal_close_deferred"
            return _finish(
                result,
                buy_cash,
                close_price,
                trade_date,
                reason,
                "daily_close",
                fee,
                adverse_marks,
                take_profit=False,
                stop_loss=False,
            )
        if terminal_day_or_later:
            result["blocked_exit_dates"].append({"trade_date": trade_date, "reason": "down_limit_close_or_untradable"})

    result["max_adverse_path"] = min(adverse_marks) if adverse_marks else None
    if terminal_reached and result["blocked_exit_dates"]:
        result["status"] = "pending_blocked_exit"
        result["exit_tradable"] = False
    return result


def _finish(
    result: dict[str, Any],
    buy_cash: float,
    reference_price: float,
    trade_date: str | None,
    reason: str,
    source: str,
    costs: ShortExecutionCostConfig,
    adverse_marks: list[float],
    *,
    take_profit: bool,
    stop_loss: bool,
    gap_through: bool = False,
) -> dict[str, Any]:
    execution_price = reference_price * (1.0 - costs.slippage_rate)
    sell_cash = execution_price * (1.0 - costs.sell_commission_rate - costs.sell_stamp_tax_rate)
    result.update(
        status="settled",
        settled=True,
        realized_net_return=sell_cash / buy_cash - 1.0,
        exit_date=trade_date,
        exit_reason=reason,
        exit_price_source=source,
        exit_reference_price=reference_price,
        exit_execution_price=execution_price,
        exit_tradable=True,
        take_profit_hit=bool(result["take_profit_hit"] or take_profit),
        stop_loss_hit=bool(result["stop_loss_hit"] or stop_loss),
        tp_before_sl=True if take_profit else (False if stop_loss else None),
        gap_through_stop=bool(result["gap_through_stop"] or gap_through),
        max_adverse_path=min(adverse_marks) if adverse_marks else None,
    )
    return result


def _net_mark_return(buy_cash: float | None, price: float | None, costs: ShortExecutionCostConfig) -> float | None:
    if buy_cash is None or price is None:
        return None
    hypothetical_sell = price * (1.0 - costs.slippage_rate) * (
        1.0 - costs.sell_commission_rate - costs.sell_stamp_tax_rate
    )
    return hypothetical_sell / buy_cash - 1.0


def _entry_tradable(open_price: float | None, up_limit: float | None) -> bool:
    return bool(open_price is not None and up_limit is not None and open_price < up_limit - 1e-8)


def _close_sellable(close_price: float | None, down_limit: float | None) -> bool:
    return bool(close_price is not None and down_limit is not None and close_price > down_limit + 1e-8)


def _entry_reason(open_price: float | None, up_limit: float | None) -> str:
    if open_price is None:
        return "missing_open"
    if up_limit is None:
        return "missing_up_limit"
    return "limit_up_open"


def _price(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("-", "")
    return text if len(text) == 8 and text.isdigit() else None
