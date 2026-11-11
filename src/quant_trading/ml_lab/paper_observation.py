"""Research-only paper observation for already-scored A-share candidates."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import and_, func

from ..db import DailyQuote, FactorSnapshot, MlTopCandidate, StockLimitDaily, session_scope
from ..services.data_sync import DataSyncService


class PaperObservationError(RuntimeError):
    """Raised when a paper observation would violate its data contract."""


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_mark_row(candidate: dict[str, Any], quote: dict[str, Any] | None, limit: dict[str, Any] | None) -> dict[str, Any]:
    """Create a non-executable same-day mark; it is not a realized T+1 return."""
    quote = quote or {}
    limit = limit or {}
    open_price = _number(quote.get("open"))
    close_price = _number(quote.get("close"))
    up_limit = _number(limit.get("up_limit"))
    down_limit = _number(limit.get("down_limit"))

    if open_price in (None, 0):
        entry_status = "missing_open_quote"
    elif up_limit is not None and open_price >= up_limit - 1e-8:
        entry_status = "blocked_limit_up_at_open"
    else:
        entry_status = "paper_entry_observable"

    same_day_mark = round(close_price / open_price - 1.0, 6) if open_price not in (None, 0) and close_price is not None else None
    return {
        "list_type": candidate["list_type"],
        "rank": int(candidate["rank"]),
        "ts_code": candidate["ts_code"],
        "name": candidate.get("name"),
        "stable_score": _number(candidate.get("stable_score")),
        "aggressive_score": _number(candidate.get("aggressive_score")),
        "risk_tags": candidate.get("risk_tags") or [],
        "entry_status": entry_status,
        "observation_open": open_price,
        "observation_close": close_price,
        "observation_pct_chg": _number(quote.get("pct_chg")),
        "up_limit": up_limit,
        "down_limit": down_limit,
        "same_day_open_to_close_mark": same_day_mark,
        "settlement_status": "unsettled_t_plus_1",
    }


def summarize_marks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    marks = [row["same_day_open_to_close_mark"] for row in rows if row["entry_status"] == "paper_entry_observable" and row["same_day_open_to_close_mark"] is not None]
    return {
        "candidate_count": len(rows),
        "entry_status_counts": dict(sorted(Counter(row["entry_status"] for row in rows).items())),
        "same_day_mark_observation_count": len(marks),
        "same_day_open_to_close_mark_mean": round(float(pd.Series(marks).mean()), 6) if marks else None,
        "settled_return_available": False,
    }


class PaperObservationService:
    """Build review artifacts without orders, quantities, positions, or promotion."""

    def run(
        self,
        signal_date: str,
        observation_date: str,
        factor_set: str,
        model_version: str,
        output_dir: str | Path,
        foundation_start_date: str = "20240401",
    ) -> dict[str, Any]:
        universe_audit = DataSyncService(client=None).stock_universe_audit()
        foundation = DataSyncService(client=None).foundation_status(foundation_start_date, observation_date)
        if universe_audit.get("status") != "pass":
            raise PaperObservationError(f"Stock-universe audit failed: {universe_audit.get('blocking')}")
        if foundation.get("status") != "pass":
            raise PaperObservationError(f"Foundation gate failed: {foundation.get('blocking')}")

        candidates, snapshot_count = self._load_candidates(signal_date, observation_date, factor_set, model_version)
        if snapshot_count == 0:
            raise PaperObservationError(f"No {factor_set} factor snapshots found for signal_date={signal_date}.")
        if not candidates:
            raise PaperObservationError(f"No scored Top10 candidates found for signal_date={signal_date}, model_version={model_version}.")

        rows = [build_mark_row(candidate, candidate.pop("quote"), candidate.pop("limit")) for candidate in candidates]
        by_list = {name: [row for row in rows if row["list_type"] == name] for name in ("stable", "aggressive")}
        payload = {
            "task": "paper_observation",
            "policy": "research-only observation; no orders, quantities, positions, buy/sell instructions, or automatic trading actions",
            "signal_date": signal_date,
            "signal_information_cutoff": f"{signal_date} close",
            "observation_date": observation_date,
            "factor_set": factor_set,
            "model_version": model_version,
            "diagnostic_only": True,
            "promotion_allowed": False,
            "market_state_v6_live_inference": {
                "status": "not_available",
                "reason": "v6 has an OOS research report but no audited live inference artifact; it is not applied to this observation.",
            },
            "data_contract": {
                "stock_universe_audit": universe_audit,
                "foundation_gate": foundation,
                "signal_snapshot_count": snapshot_count,
                "same_day_mark_policy": "Open-to-close is a mark only. A-share T+1 settlement means no completed net return is claimed on the observation date.",
            },
            "summary": {name: summarize_marks(items) for name, items in by_list.items()},
            "candidates": rows,
        }

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f"paper_observation_{factor_set}_{signal_date}_asof_{observation_date}.json"
        csv_path = directory / f"paper_observation_{factor_set}_{signal_date}_asof_{observation_date}.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return {**payload, "report_path": str(report_path), "csv_path": str(csv_path)}

    @staticmethod
    def _load_candidates(
        signal_date: str,
        observation_date: str,
        factor_set: str,
        model_version: str,
    ) -> tuple[list[dict[str, Any]], int]:
        with session_scope() as session:
            snapshot_count = int(
                session.query(func.count())
                .select_from(FactorSnapshot)
                .filter(FactorSnapshot.trade_date == signal_date, FactorSnapshot.factor_set == factor_set)
                .scalar()
                or 0
            )
            rows = (
                session.query(MlTopCandidate, DailyQuote, StockLimitDaily)
                .outerjoin(
                    DailyQuote,
                    and_(DailyQuote.ts_code == MlTopCandidate.ts_code, DailyQuote.trade_date == observation_date),
                )
                .outerjoin(
                    StockLimitDaily,
                    and_(StockLimitDaily.ts_code == MlTopCandidate.ts_code, StockLimitDaily.trade_date == observation_date),
                )
                .filter(MlTopCandidate.trade_date == signal_date, MlTopCandidate.model_version == model_version)
                .order_by(MlTopCandidate.list_type.asc(), MlTopCandidate.rank.asc())
                .all()
            )
        candidates = []
        for candidate, quote, limit in rows:
            candidates.append(
                {
                    "list_type": candidate.list_type,
                    "rank": candidate.rank,
                    "ts_code": candidate.ts_code,
                    "name": candidate.name,
                    "stable_score": candidate.stable_score,
                    "aggressive_score": candidate.aggressive_score,
                    "risk_tags": candidate.risk_tags or [],
                    "quote": {"open": quote.open if quote else None, "close": quote.close if quote else None, "pct_chg": quote.pct_chg if quote else None},
                    "limit": {"up_limit": limit.up_limit if limit else None, "down_limit": limit.down_limit if limit else None},
                }
            )
        return candidates, snapshot_count
