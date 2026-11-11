from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, func, select, text
from sqlalchemy.dialects.postgresql import insert

from ..data.tushare_client import FINA_INDICATOR_FACTOR_NAMES
from ..db import (
    DailyBasic,
    DailyQuote,
    FactorEvaluationRun,
    FactorForwardReturn,
    FactorSnapshot,
    FactorWalkForwardFold,
    FactorWalkForwardRun,
    IndexDaily,
    MoneyflowDaily,
    Stock,
    TushareRawRecord,
    session_scope,
)
from .metrics import (
    ic_ir,
    max_drawdown,
    monotonic_quantiles,
    percentile_rank,
    rank_ic,
    safe_float,
    verdict_from_metrics,
    winsorized_zscore,
)


def _select_latest_financial_payloads(rows: list[Any]) -> dict[str, dict[str, Any]]:
    """Choose the latest eligible report and the richest deterministic revision."""
    selected: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for row in rows:
        ts_code = str(getattr(row, "ts_code", "") or "")
        if not ts_code:
            continue
        payload = dict(getattr(row, "payload", None) or {})
        end_date = str(getattr(row, "end_date", None) or payload.get("end_date") or "")
        ann_date = str(getattr(row, "ann_date", None) or payload.get("ann_date") or "")
        richness = sum(safe_float(payload.get(name)) is not None for name in FINA_INDICATOR_FACTOR_NAMES)
        updated_at = str(getattr(row, "updated_at", None) or "")
        endpoint_priority = 1 if str(getattr(row, "endpoint", "")) == "fina_indicator_vip" else 0
        priority = (end_date, ann_date, richness, updated_at, endpoint_priority)
        current = selected.get(ts_code)
        if current is None or priority > current[0]:
            selected[ts_code] = (priority, payload)
    return {ts_code: payload for ts_code, (_, payload) in selected.items()}


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
FACTOR_SET_V1 = "factor_lab_v1"
FACTOR_SET_V2 = "factor_lab_v2"
FACTOR_SET_V2_FINANCIAL_V2 = "factor_lab_v2_financial_v2"
FACTOR_SET_V3 = "factor_lab_v3"
FACTOR_SET_V4 = "factor_lab_v4"
FACTOR_SET_V5 = "factor_lab_v5"
EXPANDED_FACTOR_SETS = {
    FACTOR_SET_V2,
    FACTOR_SET_V2_FINANCIAL_V2,
    FACTOR_SET_V3,
    FACTOR_SET_V4,
    FACTOR_SET_V5,
}
FACTOR_SET_V3_EXCLUDED_FACTORS = {
    "assets_yoy",
    "current_ratio",
    "debt_to_assets",
    "dv_ttm",
    "eqt_yoy",
    "free_share",
    "ocf_to_profit",
    "ocf_yoy",
    "ps_ttm",
    "pe_ttm",
    "q_op_qoq",
    "q_profit_qoq",
    "q_profit_yoy",
    "q_sales_yoy",
    "q_gsprofit_margin",
    "q_netprofit_margin",
    "q_roe",
    "quick_ratio",
    "rd_exp",
    "roa",
    "roic",
    "roe_dt",
    "roe_waa",
    "tr_yoy",
    "turnover_rate_f",
}
FACTOR_SET_V4_DROPPED_FACTORS = {
    "cross_industry_strength_flow_5d",
    "cross_quality_value",
    "cross_return_5d_main_net_5d",
    "dt_netprofit_yoy",
    "grossprofit_margin",
    "growth_composite",
    "netprofit_margin",
    "netprofit_yoy",
    "or_yoy",
    "quality_composite",
    "roe",
}
FACTOR_SET_V4_EXCLUDED_FACTORS = FACTOR_SET_V3_EXCLUDED_FACTORS | FACTOR_SET_V4_DROPPED_FACTORS
CORE_MARKET_INDEX = "000300.SH"


@dataclass(frozen=True)
class FactorLabConfig:
    factor_set: str = FACTOR_SET_V1
    fee_rate: float = 0.0003
    slippage: float = 0.0005
    min_amount: float = 30000.0
    min_listing_days: int = 120
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


class FactorLabError(RuntimeError):
    """Raised when a factor research task cannot be completed with real data."""


@dataclass
class FactorLabService:
    config: FactorLabConfig = FactorLabConfig()

    def build_snapshot(self, trade_date: str) -> dict:
        rows = self._load_snapshot_frame(trade_date)
        if rows.empty:
            raise FactorLabError(f"No local full-market quote/basic data for trade_date={trade_date}. Sync Tushare data first.")

        enriched = self._compute_cross_sectional_factors(rows, trade_date)
        if enriched.empty:
            raise FactorLabError(f"No valid factors could be computed for trade_date={trade_date}.")

        market_regime = self._market_regime(enriched)
        persisted = self._persist_snapshots(enriched, trade_date, market_regime)
        return {
            "task": "factor_snapshot",
            "trade_date": trade_date,
            "factor_set": self.config.factor_set,
            "rows": persisted,
            "market_regime": market_regime,
            "factors": self.factor_names(),
        }

    def _load_snapshot_frame(self, trade_date: str) -> pd.DataFrame:
        with session_scope() as session:
            quote_rows = (
                session.query(
                    DailyQuote.ts_code,
                    DailyQuote.trade_date,
                    DailyQuote.close,
                    DailyQuote.pre_close,
                    DailyQuote.high,
                    DailyQuote.low,
                    DailyQuote.pct_chg,
                    DailyQuote.amount,
                    Stock.name,
                    Stock.industry,
                    Stock.list_date,
                    DailyBasic.turnover_rate,
                    DailyBasic.turnover_rate_f,
                    DailyBasic.volume_ratio,
                    DailyBasic.pe_ttm,
                    DailyBasic.pb,
                    DailyBasic.ps_ttm,
                    DailyBasic.dv_ttm,
                    DailyBasic.total_mv,
                    DailyBasic.circ_mv,
                    DailyBasic.free_share,
                )
                .join(Stock, Stock.ts_code == DailyQuote.ts_code)
                .outerjoin(DailyBasic, and_(DailyBasic.ts_code == DailyQuote.ts_code, DailyBasic.trade_date == DailyQuote.trade_date))
                .filter(DailyQuote.trade_date == trade_date, Stock.list_status.in_(("L", "D")))
                .all()
            )
            moneyflow_rows = (
                session.query(
                    MoneyflowDaily.ts_code,
                    MoneyflowDaily.trade_date,
                    MoneyflowDaily.buy_sm_amount,
                    MoneyflowDaily.sell_sm_amount,
                    MoneyflowDaily.buy_md_amount,
                    MoneyflowDaily.sell_md_amount,
                    MoneyflowDaily.buy_lg_amount,
                    MoneyflowDaily.sell_lg_amount,
                    MoneyflowDaily.buy_elg_amount,
                    MoneyflowDaily.sell_elg_amount,
                    MoneyflowDaily.net_mf_amount,
                )
                .filter(MoneyflowDaily.trade_date == trade_date)
                .all()
            )

        latest_moneyflow = {}
        for row in moneyflow_rows:
            latest_moneyflow.setdefault(row.ts_code, row)

        records: list[dict[str, Any]] = []
        for row in quote_rows:
            mf = latest_moneyflow.get(row.ts_code)
            records.append(
                {
                    "ts_code": row.ts_code,
                    "trade_date": row.trade_date,
                    "name": row.name,
                    "industry": row.industry,
                    "list_date": row.list_date,
                    "close": row.close,
                    "pre_close": row.pre_close,
                    "high": row.high,
                    "low": row.low,
                    "pct_chg": row.pct_chg,
                    "amount": row.amount,
                    "turnover_rate": row.turnover_rate,
                    "turnover_rate_f": row.turnover_rate_f,
                    "volume_ratio": row.volume_ratio,
                    "pe_ttm": row.pe_ttm,
                    "pb": row.pb,
                    "ps_ttm": row.ps_ttm,
                    "dv_ttm": row.dv_ttm,
                    "total_mv": row.total_mv,
                    "circ_mv": row.circ_mv,
                    "free_share": row.free_share,
                    "buy_sm_amount": mf.buy_sm_amount if mf else None,
                    "sell_sm_amount": mf.sell_sm_amount if mf else None,
                    "buy_md_amount": mf.buy_md_amount if mf else None,
                    "sell_md_amount": mf.sell_md_amount if mf else None,
                    "buy_lg_amount": mf.buy_lg_amount if mf else None,
                    "sell_lg_amount": mf.sell_lg_amount if mf else None,
                    "buy_elg_amount": mf.buy_elg_amount if mf else None,
                    "sell_elg_amount": mf.sell_elg_amount if mf else None,
                    "net_mf_amount": mf.net_mf_amount if mf else None,
                    "moneyflow_trade_date": mf.trade_date if mf else None,
                }
            )
        return pd.DataFrame(records)

    def _compute_cross_sectional_factors(self, rows: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        uses_expanded_features = self.config.factor_set in EXPANDED_FACTOR_SETS
        history = self._load_quote_history(trade_date, lookback=170 if uses_expanded_features else 80)
        if history.empty:
            return pd.DataFrame()

        df = rows.copy()
        metrics = self._history_factor_frame_v2(history) if uses_expanded_features else self._history_factor_frame(history)
        df = df.merge(metrics, how="left", left_on="ts_code", right_index=True)
        if uses_expanded_features:
            moneyflow_metrics = self._moneyflow_factor_frame(trade_date, lookback=20)
            if not moneyflow_metrics.empty:
                df = df.merge(moneyflow_metrics, how="left", left_on="ts_code", right_index=True)
            finance_metrics = self._latest_financial_factor_frame(trade_date)
            if not finance_metrics.empty:
                df = df.merge(finance_metrics, how="left", left_on="ts_code", right_index=True)
            market_metrics = self._market_factor_payload(trade_date)
            for key, value in market_metrics.items():
                df[key] = value
        df["listing_days"] = self._listing_days_series(df["list_date"], trade_date)
        df["main_net_amount"] = self._main_net_amount_series(df)
        amount = pd.to_numeric(df["amount"], errors="coerce")
        circ_mv = pd.to_numeric(df["circ_mv"], errors="coerce")
        df["main_net_to_amount"] = df["main_net_amount"] / amount.where(amount != 0)
        df["main_net_to_circ_mv"] = df["main_net_amount"] / circ_mv.where(circ_mv != 0)
        if uses_expanded_features:
            for window in (3, 5, 10):
                sum_col = f"main_net_amount_sum_{window}d"
                if sum_col in df.columns:
                    flow_sum = pd.to_numeric(df[sum_col], errors="coerce")
                    df[f"main_net_to_amount_sum_{window}d"] = flow_sum / amount.where(amount != 0)
                    df[f"main_net_to_circ_mv_sum_{window}d"] = flow_sum / circ_mv.where(circ_mv != 0)
        if self.config.factor_set in {FACTOR_SET_V4, FACTOR_SET_V5}:
            self._add_v4_meta_factors(df)
        industry_return_20d = df.groupby("industry", dropna=True)["return_20d"].median()
        df["relative_industry_strength_20d"] = df["return_20d"] - df["industry"].map(industry_return_20d)
        if uses_expanded_features:
            for window in (5, 20, 60):
                col = f"return_{window}d"
                if col in df.columns:
                    industry_return = df.groupby("industry", dropna=True)[col].median()
                    df[f"relative_industry_strength_{window}d"] = df[col] - df["industry"].map(industry_return)
                    df[f"industry_rank_pct_{window}d"] = df.groupby("industry", dropna=False)[col].rank(pct=True)
            quality_cols = [col for col in ("roe", "roa", "grossprofit_margin", "ocf_to_profit") if col in df.columns]
            growth_cols = [col for col in ("netprofit_yoy", "dt_netprofit_yoy", "tr_yoy", "or_yoy", "ocf_yoy") if col in df.columns]
            value_cols = [col for col in ("pb", "pe_ttm", "ps_ttm") if col in df.columns]
            if quality_cols:
                df["quality_composite"] = df[quality_cols].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
            if growth_cols:
                df["growth_composite"] = df[growth_cols].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
            if value_cols:
                df["value_composite"] = 1.0 - df[value_cols].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
            if {"return_5d", "main_net_to_amount_sum_5d"}.issubset(df.columns):
                df["cross_return_5d_main_net_5d"] = pd.to_numeric(df["return_5d"], errors="coerce") * pd.to_numeric(df["main_net_to_amount_sum_5d"], errors="coerce")
            if {"relative_industry_strength_20d", "main_net_to_circ_mv_sum_5d"}.issubset(df.columns):
                df["cross_industry_strength_flow_5d"] = pd.to_numeric(df["relative_industry_strength_20d"], errors="coerce") * pd.to_numeric(df["main_net_to_circ_mv_sum_5d"], errors="coerce")
            if {"quality_composite", "value_composite"}.issubset(df.columns):
                df["cross_quality_value"] = pd.to_numeric(df["quality_composite"], errors="coerce") * pd.to_numeric(df["value_composite"], errors="coerce")
            if self.config.factor_set in {FACTOR_SET_V4, FACTOR_SET_V5}:
                self._add_v4_cross_factors(df)
            if self.config.factor_set == FACTOR_SET_V5:
                self._add_v5_cross_sectional_factors(df)
        factor_names = self.factor_names()
        missing_factor_columns = [factor for factor in factor_names if factor not in df.columns]
        if missing_factor_columns:
            df = pd.concat(
                [df, pd.DataFrame({factor: None for factor in missing_factor_columns}, index=df.index)],
                axis=1,
            )
        df["raw_factors"] = [
            {name: safe_float(record.get(name)) for name in factor_names}
            for record in df[factor_names].to_dict("records")
        ]
        normalized_columns = {}
        for factor in factor_names:
            df[factor] = df["raw_factors"].map(lambda item, key=factor: item.get(key))
            normalized_columns[f"{factor}_z"] = winsorized_zscore(df[factor])
            normalized_columns[f"{factor}_pct"] = percentile_rank(df[factor], higher_is_better=self._higher_is_better(factor))
        df = pd.concat([df, pd.DataFrame(normalized_columns, index=df.index)], axis=1).copy()

        if "circ_mv" in df.columns:
            df["market_cap_bucket"] = self._market_cap_bucket(df["circ_mv"])
        else:
            df["market_cap_bucket"] = None
        return df

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce")
        return num / den.where(den != 0)

    @staticmethod
    def _add_v4_meta_factors(df: pd.DataFrame) -> None:
        if {"return_5d", "return_20d"}.issubset(df.columns):
            df["short_vs_medium_return"] = pd.to_numeric(df["return_5d"], errors="coerce") - pd.to_numeric(df["return_20d"], errors="coerce")
        if {"return_20d", "return_60d"}.issubset(df.columns):
            df["medium_vs_long_return"] = pd.to_numeric(df["return_20d"], errors="coerce") - pd.to_numeric(df["return_60d"], errors="coerce")
        if {"volatility_5d", "volatility_20d"}.issubset(df.columns):
            df["volatility_change_5d_20d"] = FactorLabService._safe_divide(df["volatility_5d"], df["volatility_20d"]) - 1.0
        if {"amount_avg_5d", "amount_avg_20d"}.issubset(df.columns):
            df["amount_trend_5d_20d"] = FactorLabService._safe_divide(df["amount_avg_5d"], df["amount_avg_20d"]) - 1.0
        if {"distance_to_20d_high", "distance_to_20d_low"}.issubset(df.columns):
            df["range_pressure_20d"] = pd.to_numeric(df["distance_to_20d_high"], errors="coerce") + pd.to_numeric(df["distance_to_20d_low"], errors="coerce")
        if {"limit_up_count_5d", "limit_down_count_5d"}.issubset(df.columns):
            df["limit_pressure_5d"] = pd.to_numeric(df["limit_up_count_5d"], errors="coerce") - pd.to_numeric(df["limit_down_count_5d"], errors="coerce")
        if {"moneyflow_positive_days_5d", "moneyflow_positive_days_10d"}.issubset(df.columns):
            df["moneyflow_persistence_5d_10d"] = FactorLabService._safe_divide(df["moneyflow_positive_days_5d"], df["moneyflow_positive_days_10d"])

    @staticmethod
    def _add_v4_cross_factors(df: pd.DataFrame) -> None:
        if {"relative_industry_strength_20d", "volatility_20d"}.issubset(df.columns):
            df["industry_strength_risk_adjusted"] = FactorLabService._safe_divide(df["relative_industry_strength_20d"], df["volatility_20d"])
        if {"main_net_to_amount_sum_5d", "volatility_20d"}.issubset(df.columns):
            df["flow_risk_adjusted_5d"] = FactorLabService._safe_divide(df["main_net_to_amount_sum_5d"], df["volatility_20d"])
        if {"main_net_to_circ_mv_sum_5d", "return_20d"}.issubset(df.columns):
            df["flow_momentum_alignment_5d_20d"] = pd.to_numeric(df["main_net_to_circ_mv_sum_5d"], errors="coerce") * pd.to_numeric(df["return_20d"], errors="coerce")
        if {"value_composite", "volatility_20d"}.issubset(df.columns):
            df["value_risk_adjusted"] = FactorLabService._safe_divide(df["value_composite"], df["volatility_20d"])

    @staticmethod
    def _add_v5_cross_sectional_factors(df: pd.DataFrame) -> None:
        """Add same-day full-market ranks and breadth using only observed inputs."""
        rank_sources = {
            "cs_rank_return_5d": "return_5d",
            "cs_rank_return_20d": "return_20d",
            "cs_rank_relative_industry_strength_20d": "relative_industry_strength_20d",
            "cs_rank_main_net_to_amount_sum_5d": "main_net_to_amount_sum_5d",
            "cs_rank_main_net_to_circ_mv_sum_5d": "main_net_to_circ_mv_sum_5d",
            "cs_rank_volume_ratio": "volume_ratio",
            "cs_rank_turnover_rate": "turnover_rate",
            "cs_rank_volatility_20d": "volatility_20d",
            "cs_rank_amount_trend_5d_20d": "amount_trend_5d_20d",
            "cs_rank_limit_pressure_5d": "limit_pressure_5d",
        }
        for target, source in rank_sources.items():
            if source in df.columns:
                df[target] = pd.to_numeric(df[source], errors="coerce").rank(pct=True)

        if "return_5d" in df.columns:
            returns = pd.to_numeric(df["return_5d"], errors="coerce")
            df["market_breadth_return_5d"] = returns.gt(0).where(returns.notna()).mean()
            df["market_return_dispersion_5d"] = returns.std()
        if "main_net_to_amount_sum_5d" in df.columns:
            flows = pd.to_numeric(df["main_net_to_amount_sum_5d"], errors="coerce")
            df["market_breadth_flow_5d"] = flows.gt(0).where(flows.notna()).mean()
        if {"relative_industry_strength_20d", "relative_industry_strength_5d"}.issubset(df.columns):
            df["relative_strength_momentum_alignment"] = (
                pd.to_numeric(df["relative_industry_strength_20d"], errors="coerce")
                * pd.to_numeric(df["relative_industry_strength_5d"], errors="coerce")
            )
        if {"main_net_to_amount_sum_5d", "moneyflow_persistence_5d_10d"}.issubset(df.columns):
            df["flow_persistence_intensity"] = (
                pd.to_numeric(df["main_net_to_amount_sum_5d"], errors="coerce")
                * pd.to_numeric(df["moneyflow_persistence_5d_10d"], errors="coerce")
            )
        if {"return_20d", "downside_volatility_20d"}.issubset(df.columns):
            df["downside_protected_momentum_20d"] = FactorLabService._safe_divide(
                df["return_20d"], df["downside_volatility_20d"]
            )

    @staticmethod
    def _history_factor_frame(history: pd.DataFrame) -> pd.DataFrame:
        history = history.sort_values(["ts_code", "trade_date"]).copy()
        for column in ["close", "high", "low", "pct_chg"]:
            history[f"{column}_num"] = pd.to_numeric(history[column], errors="coerce")
        grouped = history.groupby("ts_code", sort=False)
        for days in (5, 20, 60):
            history[f"close_lag_{days}"] = grouped["close_num"].shift(days)
            history[f"return_{days}d"] = history["close_num"] / history[f"close_lag_{days}"] - 1.0
        history["high_20d"] = grouped["high_num"].rolling(20, min_periods=1).max().reset_index(level=0, drop=True)
        history["low_20d"] = grouped["low_num"].rolling(20, min_periods=1).min().reset_index(level=0, drop=True)
        history["mean_close_20d"] = grouped["close_num"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
        history["volatility_20d"] = (grouped["pct_chg_num"].rolling(20, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)) / 100.0
        latest = history.groupby("ts_code", sort=False).tail(1).set_index("ts_code")
        result = pd.DataFrame(index=latest.index)
        result["return_5d"] = latest["return_5d"]
        result["return_20d"] = latest["return_20d"]
        result["return_60d"] = latest["return_60d"]
        result["distance_to_20d_high"] = latest["close_num"] / latest["high_20d"].where(latest["high_20d"] != 0) - 1.0
        result["drawdown_20d"] = result["distance_to_20d_high"]
        result["volatility_20d"] = latest["volatility_20d"]
        result["amplitude_20d"] = (latest["high_20d"] - latest["low_20d"]) / latest["mean_close_20d"].where(latest["mean_close_20d"] != 0)
        return result

    @staticmethod
    def _history_factor_frame_v2(history: pd.DataFrame) -> pd.DataFrame:
        history = history.sort_values(["ts_code", "trade_date"]).copy()
        for column in ["close", "high", "low", "pct_chg", "amount"]:
            history[f"{column}_num"] = pd.to_numeric(history[column], errors="coerce")
        grouped = history.groupby("ts_code", sort=False)
        for days in (1, 2, 3, 5, 10, 20, 60, 120):
            history[f"close_lag_{days}"] = grouped["close_num"].shift(days)
            history[f"return_{days}d"] = history["close_num"] / history[f"close_lag_{days}"] - 1.0
        for days in (5, 10, 20, 60, 120):
            history[f"high_{days}d"] = grouped["high_num"].rolling(days, min_periods=1).max().reset_index(level=0, drop=True)
            history[f"low_{days}d"] = grouped["low_num"].rolling(days, min_periods=1).min().reset_index(level=0, drop=True)
            history[f"ma_{days}d"] = grouped["close_num"].rolling(days, min_periods=1).mean().reset_index(level=0, drop=True)
            history[f"amount_avg_{days}d"] = grouped["amount_num"].rolling(days, min_periods=1).mean().reset_index(level=0, drop=True)
            history[f"volatility_{days}d"] = grouped["pct_chg_num"].rolling(days, min_periods=1).std(ddof=0).reset_index(level=0, drop=True) / 100.0
        history["downside_pct_20d"] = history["pct_chg_num"].where(history["pct_chg_num"] < 0)
        history["downside_volatility_20d"] = grouped["downside_pct_20d"].rolling(20, min_periods=1).std(ddof=0).reset_index(level=0, drop=True) / 100.0
        history["limit_up_flag"] = (history["pct_chg_num"] >= 9.8).astype(float)
        history["limit_down_flag"] = (history["pct_chg_num"] <= -9.8).astype(float)
        for days in (5, 10, 20):
            history[f"limit_up_count_{days}d"] = grouped["limit_up_flag"].rolling(days, min_periods=1).sum().reset_index(level=0, drop=True)
            history[f"limit_down_count_{days}d"] = grouped["limit_down_flag"].rolling(days, min_periods=1).sum().reset_index(level=0, drop=True)
        history["abs_return_amount"] = (history["pct_chg_num"].abs() / 100.0) / history["amount_num"].where(history["amount_num"] != 0)
        history["amihud_illiquidity_20d"] = grouped["abs_return_amount"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
        latest = history.groupby("ts_code", sort=False).tail(1).set_index("ts_code")
        result = pd.DataFrame(index=latest.index)
        for days in (1, 2, 3, 5, 10, 20, 60, 120):
            result[f"return_{days}d"] = latest[f"return_{days}d"]
        for days in (1, 3, 5):
            result[f"reversal_{days}d"] = -latest[f"return_{days}d"]
        for days in (5, 10, 20, 60):
            result[f"ma_gap_{days}d"] = latest["close_num"] / latest[f"ma_{days}d"].where(latest[f"ma_{days}d"] != 0) - 1.0
            lag = history.groupby("ts_code", sort=False)[f"ma_{days}d"].shift(days).groupby(history["ts_code"], sort=False).tail(1).values
            result[f"ma_slope_{days}d"] = latest[f"ma_{days}d"].values / pd.Series(lag, index=latest.index).where(pd.Series(lag, index=latest.index) != 0) - 1.0
        for days in (20, 60, 120):
            result[f"distance_to_{days}d_high"] = latest["close_num"] / latest[f"high_{days}d"].where(latest[f"high_{days}d"] != 0) - 1.0
        for days in (5, 20, 60):
            result[f"distance_to_{days}d_low"] = latest["close_num"] / latest[f"low_{days}d"].where(latest[f"low_{days}d"] != 0) - 1.0
            result[f"amplitude_{days}d"] = (latest[f"high_{days}d"] - latest[f"low_{days}d"]) / latest[f"ma_{days}d"].where(latest[f"ma_{days}d"] != 0)
        result["distance_to_20d_high"] = result["distance_to_20d_high"]
        result["drawdown_20d"] = result["distance_to_20d_high"]
        for days in (5, 10, 20, 60):
            result[f"volatility_{days}d"] = latest[f"volatility_{days}d"]
            result[f"amount_avg_{days}d"] = latest[f"amount_avg_{days}d"]
        result["volatility_20d"] = result["volatility_20d"]
        result["amplitude_20d"] = result["amplitude_20d"]
        result["downside_volatility_20d"] = latest["downside_volatility_20d"]
        result["amihud_illiquidity_20d"] = latest["amihud_illiquidity_20d"]
        for days in (5, 10, 20):
            result[f"limit_up_count_{days}d"] = latest[f"limit_up_count_{days}d"]
            result[f"limit_down_count_{days}d"] = latest[f"limit_down_count_{days}d"]
        result["close_position_in_range_20d"] = (latest["close_num"] - latest["low_20d"]) / (latest["high_20d"] - latest["low_20d"]).where((latest["high_20d"] - latest["low_20d"]) != 0)
        result["amount_zscore_20d"] = (latest["amount_num"] - latest["amount_avg_20d"]) / grouped["amount_num"].rolling(20, min_periods=1).std(ddof=0).reset_index(level=0, drop=True).groupby(history["ts_code"], sort=False).tail(1).set_axis(latest.index).replace(0, pd.NA)
        return result

    def _load_quote_history(self, trade_date: str, lookback: int) -> pd.DataFrame:
        start_date = self._lookback_start_date(trade_date, lookback + 10)
        with session_scope() as session:
            rows = (
                session.query(
                    DailyQuote.ts_code,
                    DailyQuote.trade_date,
                    DailyQuote.close,
                    DailyQuote.high,
                    DailyQuote.low,
                    DailyQuote.pct_chg,
                    DailyQuote.amount,
                )
                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= trade_date)
                .order_by(DailyQuote.ts_code.asc(), DailyQuote.trade_date.desc())
                .all()
            )
        records = [
            {
                "ts_code": row.ts_code,
                "trade_date": row.trade_date,
                "close": row.close,
                "high": row.high,
                "low": row.low,
                "pct_chg": row.pct_chg,
                "amount": row.amount,
            }
            for row in rows
        ]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.groupby("ts_code", group_keys=False).head(lookback)

    def _moneyflow_factor_frame(self, trade_date: str, lookback: int) -> pd.DataFrame:
        history = self._load_moneyflow_history(trade_date, lookback)
        if history.empty:
            return pd.DataFrame()
        history = history.sort_values(["ts_code", "trade_date"]).copy()
        for column in ["buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]:
            history[column] = pd.to_numeric(history[column], errors="coerce")
        history["main_net_amount_v2"] = history["net_mf_amount"]
        missing_net = history["main_net_amount_v2"].isna()
        component_cols = ["buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]
        has_components = history[component_cols].notna().any(axis=1)
        fallback_net = (
            history["buy_lg_amount"].fillna(0.0)
            + history["buy_elg_amount"].fillna(0.0)
            - history["sell_lg_amount"].fillna(0.0)
            - history["sell_elg_amount"].fillna(0.0)
        )
        history.loc[missing_net & has_components, "main_net_amount_v2"] = fallback_net[missing_net & has_components]
        history["large_order_net_amount"] = history["buy_lg_amount"].fillna(0.0) - history["sell_lg_amount"].fillna(0.0)
        history.loc[history[["buy_lg_amount", "sell_lg_amount"]].isna().all(axis=1), "large_order_net_amount"] = pd.NA
        history["extra_large_order_net_amount"] = history["buy_elg_amount"].fillna(0.0) - history["sell_elg_amount"].fillna(0.0)
        history.loc[history[["buy_elg_amount", "sell_elg_amount"]].isna().all(axis=1), "extra_large_order_net_amount"] = pd.NA
        grouped = history.groupby("ts_code", sort=False)
        latest = history.groupby("ts_code", sort=False).tail(1).set_index("ts_code")
        result = pd.DataFrame(index=latest.index)
        for days in (3, 5, 10):
            result[f"main_net_amount_sum_{days}d"] = grouped["main_net_amount_v2"].rolling(days, min_periods=1).sum().reset_index(level=0, drop=True).groupby(history["ts_code"], sort=False).tail(1).set_axis(latest.index)
            result[f"large_order_net_sum_{days}d"] = grouped["large_order_net_amount"].rolling(days, min_periods=1).sum().reset_index(level=0, drop=True).groupby(history["ts_code"], sort=False).tail(1).set_axis(latest.index)
            result[f"extra_large_order_net_sum_{days}d"] = grouped["extra_large_order_net_amount"].rolling(days, min_periods=1).sum().reset_index(level=0, drop=True).groupby(history["ts_code"], sort=False).tail(1).set_axis(latest.index)
            result[f"moneyflow_positive_days_{days}d"] = grouped["main_net_amount_v2"].rolling(days, min_periods=1).apply(lambda values: float((values > 0).sum()), raw=False).reset_index(level=0, drop=True).groupby(history["ts_code"], sort=False).tail(1).set_axis(latest.index)
        return result

    def _load_moneyflow_history(self, trade_date: str, lookback: int) -> pd.DataFrame:
        start_date = self._lookback_start_date(trade_date, lookback + 10)
        with session_scope() as session:
            rows = (
                session.query(
                    MoneyflowDaily.ts_code,
                    MoneyflowDaily.trade_date,
                    MoneyflowDaily.buy_sm_amount,
                    MoneyflowDaily.sell_sm_amount,
                    MoneyflowDaily.buy_md_amount,
                    MoneyflowDaily.sell_md_amount,
                    MoneyflowDaily.buy_lg_amount,
                    MoneyflowDaily.sell_lg_amount,
                    MoneyflowDaily.buy_elg_amount,
                    MoneyflowDaily.sell_elg_amount,
                    MoneyflowDaily.net_mf_amount,
                )
                .filter(MoneyflowDaily.trade_date >= start_date, MoneyflowDaily.trade_date <= trade_date)
                .order_by(MoneyflowDaily.ts_code.asc(), MoneyflowDaily.trade_date.desc())
                .all()
            )
        records = [row._asdict() for row in rows]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.groupby("ts_code", group_keys=False).head(lookback)

    def _latest_financial_factor_frame(self, trade_date: str) -> pd.DataFrame:
        endpoints = ["fina_indicator", "fina_indicator_vip"]
        with session_scope() as session:
            rows = (
                session.query(TushareRawRecord)
                # Date-only announcements have no disclosure time. Excluding the
                # signal day prevents after-close disclosures leaking into D close.
                .filter(TushareRawRecord.endpoint.in_(endpoints), TushareRawRecord.ann_date < trade_date)
                .all()
            )
        latest = _select_latest_financial_payloads(rows)
        if not latest:
            return self._latest_legacy_fina_indicator_frame(trade_date)
        records = []
        for ts_code, payload in latest.items():
            item = {"ts_code": ts_code}
            for field in FINA_INDICATOR_FACTOR_NAMES:
                item[field] = safe_float(payload.get(field))
            records.append(item)
        return pd.DataFrame(records).set_index("ts_code")

    def _latest_legacy_fina_indicator_frame(self, trade_date: str) -> pd.DataFrame:
        fields = list(FINA_INDICATOR_FACTOR_NAMES)
        with session_scope() as session:
            exists = session.execute(text("select to_regclass('public.raw_fina_indicator')")).scalar()
            if exists is None:
                return pd.DataFrame()
            columns = {
                row[0]
                for row in session.execute(
                    text(
                        "select column_name from information_schema.columns "
                        "where table_schema='public' and table_name='raw_fina_indicator'"
                    )
                )
            }
            select_fields = []
            for field in fields:
                if field in columns:
                    select_fields.append(field)
                else:
                    select_fields.append(f"NULL::double precision AS {field}")
            sql = (
                "select distinct on (ts_code) ts_code, "
                + ", ".join(select_fields)
                + " from raw_fina_indicator "
                + "where ann_date < :trade_date "
                + "order by ts_code asc, ann_date desc, end_date desc"
            )
            rows = session.execute(text(sql), {"trade_date": trade_date}).mappings().all()
        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            item = {"ts_code": row["ts_code"]}
            for field in fields:
                item[field] = safe_float(row.get(field))
            records.append(item)
        return pd.DataFrame(records).set_index("ts_code")

    def _market_factor_payload(self, trade_date: str) -> dict[str, float | None]:
        start_date = self._lookback_start_date(trade_date, 80)
        with session_scope() as session:
            rows = (
                session.query(IndexDaily)
                .filter(IndexDaily.ts_code == CORE_MARKET_INDEX, IndexDaily.trade_date >= start_date, IndexDaily.trade_date <= trade_date)
                .order_by(IndexDaily.trade_date.asc())
                .all()
            )
        if not rows:
            return {
                "market_return_1d": None,
                "market_return_5d": None,
                "market_return_20d": None,
                "market_volatility_20d": None,
                "market_drawdown_20d": None,
            }
        df = pd.DataFrame(
            [{"trade_date": row.trade_date, "close": row.close, "pct_chg": row.pct_chg} for row in rows]
        ).sort_values("trade_date")
        close = pd.to_numeric(df["close"], errors="coerce")
        pct = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
        high_20 = close.tail(20).max()
        latest = close.iloc[-1]
        return {
            "market_return_1d": safe_float(close.iloc[-1] / close.iloc[-2] - 1.0) if len(close) >= 2 and close.iloc[-2] else None,
            "market_return_5d": safe_float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 and close.iloc[-6] else None,
            "market_return_20d": safe_float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 and close.iloc[-21] else None,
            "market_volatility_20d": safe_float(pct.tail(20).std(ddof=0)),
            "market_drawdown_20d": safe_float(latest / high_20 - 1.0) if high_20 else None,
        }

    @staticmethod
    def _lookback_start_date(trade_date: str, lookback: int) -> str:
        with session_scope() as session:
            rows = (
                session.query(DailyQuote.trade_date)
                .filter(DailyQuote.trade_date <= trade_date)
                .group_by(DailyQuote.trade_date)
                .order_by(DailyQuote.trade_date.desc())
                .limit(lookback)
                .all()
            )
        if not rows:
            return trade_date
        return rows[-1][0]

    def _raw_factors_for_stock(self, row: dict[str, Any], history: pd.DataFrame, trade_date: str) -> dict:
        history = history.sort_values("trade_date").copy()
        close = pd.to_numeric(history["close"], errors="coerce")
        high = pd.to_numeric(history["high"], errors="coerce")
        low = pd.to_numeric(history["low"], errors="coerce")
        pct = pd.to_numeric(history["pct_chg"], errors="coerce") / 100.0
        amount = safe_float(row.get("amount"))
        circ_mv = safe_float(row.get("circ_mv"))
        main_net = self._main_net_amount(row)

        raw = {
            "amount": amount,
            "turnover_rate": safe_float(row.get("turnover_rate")),
            "volume_ratio": safe_float(row.get("volume_ratio")),
            "circ_mv": circ_mv,
            "listing_days": self._listing_days(row.get("list_date"), trade_date),
            "return_5d": self._period_return(close, 5),
            "return_20d": self._period_return(close, 20),
            "return_60d": self._period_return(close, 60),
            "distance_to_20d_high": self._distance_to_high(close, high, 20),
            "drawdown_20d": self._drawdown(close, high, 20),
            "volatility_20d": safe_float(pct.tail(20).std(ddof=0)),
            "amplitude_20d": self._amplitude(high, low, close, 20),
            "pe_ttm": safe_float(row.get("pe_ttm")),
            "pb": safe_float(row.get("pb")),
            "main_net_amount": main_net,
            "main_net_to_amount": main_net / amount if main_net is not None and amount not in (None, 0) else None,
            "main_net_to_circ_mv": main_net / circ_mv if main_net is not None and circ_mv not in (None, 0) else None,
        }
        raw["relative_industry_strength_20d"] = None
        return raw

    def build_forward_returns(self, trade_date: str, horizons: tuple[int, ...] | None = None) -> dict:
        horizons = horizons or self.config.horizons
        with session_scope() as session:
            snapshots = (
                session.query(FactorSnapshot)
                .filter(FactorSnapshot.trade_date == trade_date, FactorSnapshot.factor_set == self.config.factor_set)
                .all()
            )
        if not snapshots:
            raise FactorLabError(f"No factor snapshots found for trade_date={trade_date}.")

        persisted = 0
        for snapshot in snapshots:
            quotes = self._future_quotes(snapshot.ts_code, trade_date, max(horizons) + 2)
            for horizon in horizons:
                payload = self._forward_return_payload(snapshot.ts_code, trade_date, quotes, horizon)
                self._persist_forward_return(payload)
                persisted += 1
        return {"task": "factor_forward_returns", "trade_date": trade_date, "horizons": list(horizons), "rows": persisted}

    def evaluate_factor(self, factor_name: str, start_date: str, end_date: str, horizon: int = 5, quantiles: int = 5) -> dict:
        existing = self._existing_evaluation(factor_name, start_date, end_date, horizon)
        if existing is not None:
            return existing

        df = self._evaluation_frame(factor_name, start_date, end_date, horizon)
        if df.empty:
            raise FactorLabError(f"No evaluation data for factor={factor_name}, horizon={horizon}, range={start_date}-{end_date}.")

        ic_rows = []
        for date_value, group in df.groupby("trade_date"):
            value = rank_ic(group["factor_value"], group["net_forward_return"])
            if value is not None:
                ic_rows.append({"trade_date": date_value, "rank_ic": value})

        ic_values = [row["rank_ic"] for row in ic_rows]
        rank_ic_mean = safe_float(pd.Series(ic_values).mean()) if ic_values else None
        rank_ic_std = safe_float(pd.Series(ic_values).std(ddof=0)) if len(ic_values) > 1 else None
        rank_ic_ir = ic_ir(ic_values)

        quantile_summary = self._quantile_summary(df, quantiles)
        top_summary = self._top_summary(df, top_ns=(50, 100))
        group_summary = self._group_summary(df)
        top50_return = top_summary.get("top50", {}).get("mean_return")
        verdict = verdict_from_metrics(rank_ic_mean, rank_ic_ir, top50_return, monotonic_quantiles(quantile_summary))

        summary = {
            "factor_name": factor_name,
            "factor_set": self.config.factor_set,
            "start_date": start_date,
            "end_date": end_date,
            "horizon": horizon,
            "sample_size": int(len(df)),
            "date_count": int(df["trade_date"].nunique()),
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_ic_ir": rank_ic_ir,
            "quantile_returns": quantile_summary,
            "top_returns": top_summary,
            "group_summary": group_summary,
            "max_drawdown": max_drawdown(df.groupby("trade_date")["net_forward_return"].mean().sort_index().tolist()),
            "verdict": verdict,
        }
        run_id = self._persist_evaluation(factor_name, start_date, end_date, horizon, summary)
        return {"run_id": run_id, "summary": summary}

    def walk_forward_factor(self, factor_name: str, train_window: int, validation_window: int, step_window: int, start_date: str, end_date: str, horizon: int = 5) -> dict:
        trade_dates = self._trade_dates_between(start_date, end_date)
        if len(trade_dates) < train_window + validation_window:
            raise FactorLabError("Not enough trade dates for walk-forward.")

        folds = []
        summary = {
            "factor_name": factor_name,
            "factor_set": self.config.factor_set,
            "start_date": start_date,
            "end_date": end_date,
            "horizon": horizon,
            "train_window": train_window,
            "validation_window": validation_window,
            "step_window": step_window,
            "folds": 0,
            "validation_verdicts": [],
        }
        run_id = self._persist_walk_forward_run(factor_name, train_window, validation_window, step_window, start_date, end_date, summary)
        for fold_index, offset in enumerate(range(0, len(trade_dates) - train_window - validation_window + 1, step_window)):
            train_dates = trade_dates[offset : offset + train_window]
            valid_dates = trade_dates[offset + train_window : offset + train_window + validation_window]
            if not train_dates or not valid_dates:
                continue
            train_start, train_end = train_dates[0], train_dates[-1]
            valid_start, valid_end = valid_dates[0], valid_dates[-1]
            train_eval = self._evaluate_factor_frame(factor_name, train_start, train_end, horizon)
            valid_eval = self._evaluate_factor_frame(factor_name, valid_start, valid_end, horizon)
            fold_summary = {
                "fold_index": fold_index,
                "train": train_eval,
                "validation": valid_eval,
                "train_window": {"start_date": train_start, "end_date": train_end},
                "validation_window": {"start_date": valid_start, "end_date": valid_end},
            }
            self._persist_walk_forward_fold(
                run_id=run_id,
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                summary=fold_summary,
            )
            folds.append(fold_summary)

        summary["folds"] = len(folds)
        summary["validation_verdicts"] = [fold["validation"].get("verdict") for fold in folds]
        return {"run_id": run_id, "summary": summary, "folds": folds}

    def _evaluate_factor_frame(self, factor_name: str, start_date: str, end_date: str, horizon: int, quantiles: int = 5) -> dict:
        df = self._evaluation_frame(factor_name, start_date, end_date, horizon)
        if df.empty:
            return {
                "factor_name": factor_name,
                "factor_set": self.config.factor_set,
                "start_date": start_date,
                "end_date": end_date,
                "horizon": horizon,
                "sample_size": 0,
                "date_count": 0,
                "rank_ic_mean": None,
                "rank_ic_std": None,
                "rank_ic_ir": None,
                "quantile_returns": {},
                "top_returns": {},
                "group_summary": {},
                "max_drawdown": None,
                "verdict": "弃用",
            }

        ic_rows = []
        for date_value, group in df.groupby("trade_date"):
            value = rank_ic(group["factor_value"], group["net_forward_return"])
            if value is not None:
                ic_rows.append({"trade_date": date_value, "rank_ic": value})

        ic_values = [row["rank_ic"] for row in ic_rows]
        rank_ic_mean = safe_float(pd.Series(ic_values).mean()) if ic_values else None
        rank_ic_std = safe_float(pd.Series(ic_values).std(ddof=0)) if len(ic_values) > 1 else None
        rank_ic_ir = ic_ir(ic_values)
        quantile_summary = self._quantile_summary(df, quantiles)
        top_summary = self._top_summary(df, top_ns=(50, 100))
        group_summary = self._group_summary(df)
        top50_return = top_summary.get("top50", {}).get("mean_return")
        verdict = verdict_from_metrics(rank_ic_mean, rank_ic_ir, top50_return, monotonic_quantiles(quantile_summary))
        return {
            "factor_name": factor_name,
            "factor_set": self.config.factor_set,
            "start_date": start_date,
            "end_date": end_date,
            "horizon": horizon,
            "sample_size": int(len(df)),
            "date_count": int(df["trade_date"].nunique()),
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_ic_ir": rank_ic_ir,
            "quantile_returns": quantile_summary,
            "top_returns": top_summary,
            "group_summary": group_summary,
            "max_drawdown": max_drawdown(df.groupby("trade_date")["net_forward_return"].mean().sort_index().tolist()),
            "verdict": verdict,
        }

    def _future_quotes(self, ts_code: str, trade_date: str, limit: int) -> list[DailyQuote]:
        with session_scope() as session:
            return (
                session.query(DailyQuote)
                .filter(DailyQuote.ts_code == ts_code, DailyQuote.trade_date > trade_date)
                .order_by(DailyQuote.trade_date.asc())
                .limit(limit)
                .all()
            )

    def _forward_return_payload(self, ts_code: str, trade_date: str, quotes: list[DailyQuote], horizon: int) -> dict:
        constraints: dict[str, Any] = {}
        if len(quotes) < horizon:
            return {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "horizon": horizon,
                "forward_return": None,
                "net_forward_return": None,
                "entry_date": None,
                "exit_date": None,
                "tradable": False,
                "constraints": {"insufficient_future_quotes": True},
            }

        entry = quotes[0]
        exit_quote = quotes[horizon - 1]
        entry_price = safe_float(entry.open) or safe_float(entry.close)
        exit_price = safe_float(exit_quote.close)
        entry_pct = safe_float(entry.pct_chg)
        exit_pct = safe_float(exit_quote.pct_chg)
        if entry_price is None or exit_price is None:
            constraints["missing_price"] = True
        if entry_pct is not None and entry_pct >= 9.8:
            constraints["entry_limit_up"] = True
        if exit_pct is not None and exit_pct <= -9.8:
            constraints["exit_limit_down"] = True

        tradable = not constraints
        gross = exit_price / entry_price - 1.0 if entry_price and exit_price else None
        round_trip_cost = self.config.fee_rate * 2 + self.config.slippage * 2
        net = gross - round_trip_cost if gross is not None and tradable else None
        return {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "horizon": horizon,
            "forward_return": gross,
            "net_forward_return": net,
            "entry_date": entry.trade_date,
            "exit_date": exit_quote.trade_date,
            "tradable": tradable,
            "constraints": constraints,
        }

    def _persist_forward_return(self, payload: dict) -> None:
        with session_scope() as session:
            stmt = insert(FactorForwardReturn).values(**payload)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_forward_return_date_stock_horizon",
                set_={
                    "forward_return": stmt.excluded.forward_return,
                    "net_forward_return": stmt.excluded.net_forward_return,
                    "entry_date": stmt.excluded.entry_date,
                    "exit_date": stmt.excluded.exit_date,
                    "tradable": stmt.excluded.tradable,
                    "constraints": stmt.excluded.constraints,
                },
            )
            session.execute(stmt)

    def _evaluation_frame(self, factor_name: str, start_date: str, end_date: str, horizon: int) -> pd.DataFrame:
        factor_value = func.json_extract_path_text(FactorSnapshot.normalized_factors, factor_name).label("factor_value")
        with session_scope() as session:
            rows = (
                session.query(
                    FactorSnapshot.trade_date,
                    FactorSnapshot.ts_code,
                    FactorSnapshot.industry,
                    FactorSnapshot.market_regime,
                    FactorSnapshot.market_cap_bucket,
                    factor_value,
                    FactorForwardReturn.net_forward_return,
                )
                .join(
                    FactorForwardReturn,
                    and_(
                        FactorForwardReturn.ts_code == FactorSnapshot.ts_code,
                        FactorForwardReturn.trade_date == FactorSnapshot.trade_date,
                    ),
                )
                .filter(
                    FactorSnapshot.factor_set == self.config.factor_set,
                    FactorSnapshot.trade_date >= start_date,
                    FactorSnapshot.trade_date <= end_date,
                    FactorForwardReturn.horizon == horizon,
                    FactorForwardReturn.tradable.is_(True),
                    factor_value.isnot(None),
                    FactorForwardReturn.net_forward_return.isnot(None),
                )
                .all()
            )
        records = []
        for row in rows:
            value = safe_float(row.factor_value)
            net_return = safe_float(row.net_forward_return)
            if value is None or net_return is None:
                continue
            records.append(
                {
                    "trade_date": row.trade_date,
                    "ts_code": row.ts_code,
                    "industry": row.industry,
                    "market_regime": row.market_regime,
                    "market_cap_bucket": row.market_cap_bucket,
                    "factor_value": value,
                    "net_forward_return": net_return,
                }
            )
        return pd.DataFrame(records)

    def _quantile_summary(self, df: pd.DataFrame, quantiles: int) -> dict[str, float | None]:
        working = df.copy()
        working["quantile"] = working.groupby("trade_date")["factor_value"].transform(
            lambda col: self._safe_quantile_labels(col, quantiles)
        )
        summary = working.groupby("quantile")["net_forward_return"].mean()
        return {f"q{int(key)}": safe_float(value) for key, value in summary.items()}

    def _top_summary(self, df: pd.DataFrame, top_ns: tuple[int, ...]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for top_n in top_ns:
            top = df.sort_values(["trade_date", "factor_value"], ascending=[True, False]).groupby("trade_date").head(top_n)
            returns = top.groupby("trade_date")["net_forward_return"].mean().sort_index()
            result[f"top{top_n}"] = {
                "mean_return": safe_float(returns.mean()),
                "win_rate": safe_float((returns > 0).mean()) if not returns.empty else None,
                "max_drawdown": max_drawdown(returns.tolist()),
            }
        return result

    def _group_summary(self, df: pd.DataFrame) -> dict:
        result = {}
        for group_name in ["market_regime", "market_cap_bucket", "industry"]:
            if group_name not in df.columns:
                continue
            grouped = df.groupby(group_name, dropna=True)["net_forward_return"].agg(["count", "mean"])
            result[group_name] = {
                str(index): {"count": int(row["count"]), "mean_return": safe_float(row["mean"])}
                for index, row in grouped.iterrows()
            }
        return result

    def _persist_evaluation(self, factor_name: str, start_date: str, end_date: str, horizon: int, summary: dict) -> int:
        with session_scope() as session:
            run = FactorEvaluationRun(
                factor_set=self.config.factor_set,
                factor_name=factor_name,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                status="completed",
                summary=summary,
            )
            session.add(run)
            session.flush()
            return int(run.id)

    def _persist_walk_forward_run(self, factor_name: str, train_window: int, validation_window: int, step_window: int, start_date: str, end_date: str, summary: dict) -> int:
        with session_scope() as session:
            run = FactorWalkForwardRun(
                factor_set=self.config.factor_set,
                strategy_name=factor_name,
                start_date=start_date,
                end_date=end_date,
                train_window=train_window,
                validation_window=validation_window,
                step_window=step_window,
                summary=summary,
            )
            session.add(run)
            session.flush()
            return int(run.id)

    def _persist_walk_forward_fold(self, run_id: int | None, fold_index: int, train_start: str, train_end: str, valid_start: str, valid_end: str, summary: dict) -> None:
        if run_id is None:
            return
        with session_scope() as session:
            session.add(
                FactorWalkForwardFold(
                    run_id=run_id,
                    fold_index=fold_index,
                    train_start_date=train_start,
                    train_end_date=train_end,
                    validation_start_date=valid_start,
                    validation_end_date=valid_end,
                    summary=summary,
                )
            )

    def _trade_dates_between(self, start_date: str, end_date: str) -> list[str]:
        with session_scope() as session:
            rows = (
                session.query(DailyQuote.trade_date)
                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)
                .group_by(DailyQuote.trade_date)
                .order_by(DailyQuote.trade_date.asc())
                .all()
            )
        return [row[0] for row in rows]

    def _existing_evaluation(self, factor_name: str, start_date: str, end_date: str, horizon: int) -> dict | None:
        with session_scope() as session:
            run = (
                session.query(FactorEvaluationRun)
                .filter(
                    FactorEvaluationRun.factor_set == self.config.factor_set,
                    FactorEvaluationRun.factor_name == factor_name,
                    FactorEvaluationRun.start_date == start_date,
                    FactorEvaluationRun.end_date == end_date,
                    FactorEvaluationRun.horizon == horizon,
                    FactorEvaluationRun.status == "completed",
                )
                .order_by(FactorEvaluationRun.id.desc())
                .first()
            )
            if run is None:
                return None
            return {"run_id": int(run.id), "summary": run.summary, "skipped": True}

    def _persist_snapshots(self, df: pd.DataFrame, trade_date: str, market_regime: str) -> int:
        now = datetime.utcnow()
        payloads = []
        for row in df.to_dict("records"):
            raw = row["raw_factors"]
            normalized = {factor: safe_float(row.get(f"{factor}_pct")) for factor in self.factor_names()}
            data_quality = self._data_quality(row, raw)
            payloads.append(
                {
                    "trade_date": trade_date,
                    "ts_code": row["ts_code"],
                    "factor_set": self.config.factor_set,
                    "industry": row.get("industry"),
                    "market_regime": market_regime,
                    "market_cap_bucket": row.get("market_cap_bucket"),
                    "raw_factors": _json_clean(raw),
                    "normalized_factors": _json_clean(normalized),
                    "data_quality": _json_clean(data_quality),
                    "created_at": now,
                }
            )
        if not payloads:
            return 0
        with session_scope() as session:
            for start in range(0, len(payloads), 1000):
                chunk = payloads[start : start + 1000]
                stmt = insert(FactorSnapshot).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_factor_snapshot_date_stock_set",
                    set_={
                        "industry": stmt.excluded.industry,
                        "market_regime": stmt.excluded.market_regime,
                        "market_cap_bucket": stmt.excluded.market_cap_bucket,
                        "raw_factors": stmt.excluded.raw_factors,
                        "normalized_factors": stmt.excluded.normalized_factors,
                        "data_quality": stmt.excluded.data_quality,
                    },
                )
                session.execute(stmt)
        return len(payloads)

    def _data_quality(self, row: dict, raw: dict) -> dict:
        missing = [name for name, value in raw.items() if safe_float(value) is None]
        financial_factor_names = {
            "roe",
            "roe_dt",
            "roe_waa",
            "roa",
            "roic",
            "grossprofit_margin",
            "netprofit_margin",
            "ocf_to_profit",
            "debt_to_assets",
            "current_ratio",
            "quick_ratio",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "tr_yoy",
            "or_yoy",
            "ocf_yoy",
            "q_roe",
            "q_sales_yoy",
            "q_profit_yoy",
            "q_profit_qoq",
            "q_op_qoq",
            "q_gsprofit_margin",
            "q_netprofit_margin",
            "assets_yoy",
            "eqt_yoy",
            "rd_exp",
        }
        has_financial_factors = any(safe_float(raw.get(name)) is not None for name in financial_factor_names)
        constraints = {
            "low_liquidity": (safe_float(raw.get("amount")) or 0) < self.config.min_amount,
            "short_listing": (safe_float(raw.get("listing_days")) or 0) < self.config.min_listing_days,
            "moneyflow_missing": row.get("moneyflow_trade_date") in (None, ""),
            "financial_statement_factors_deferred": not has_financial_factors,
        }
        return {"missing_factors": missing, "constraints": constraints}

    @staticmethod
    def _with_relative_industry_strength(raw: dict, industry_median: float | None) -> dict:
        copied = dict(raw)
        ret = safe_float(copied.get("return_20d"))
        copied["relative_industry_strength_20d"] = ret - industry_median if ret is not None and industry_median is not None else None
        return copied

    def factor_names(self) -> list[str]:
        return self.factor_names_for_set(self.config.factor_set)

    @staticmethod
    def factor_names_for_set(factor_set: str = FACTOR_SET_V1) -> list[str]:
        v1 = [
            "amount",
            "turnover_rate",
            "volume_ratio",
            "circ_mv",
            "listing_days",
            "return_5d",
            "return_20d",
            "return_60d",
            "distance_to_20d_high",
            "drawdown_20d",
            "volatility_20d",
            "amplitude_20d",
            "pe_ttm",
            "pb",
            "main_net_amount",
            "main_net_to_amount",
            "main_net_to_circ_mv",
            "relative_industry_strength_20d",
        ]
        if factor_set not in EXPANDED_FACTOR_SETS:
            return v1
        v2_extra = [
            "turnover_rate_f",
            "free_share",
            "ps_ttm",
            "dv_ttm",
            "return_1d",
            "return_2d",
            "return_3d",
            "return_10d",
            "return_120d",
            "reversal_1d",
            "reversal_3d",
            "reversal_5d",
            "ma_gap_5d",
            "ma_gap_10d",
            "ma_gap_20d",
            "ma_gap_60d",
            "ma_slope_5d",
            "ma_slope_10d",
            "ma_slope_20d",
            "ma_slope_60d",
            "distance_to_60d_high",
            "distance_to_120d_high",
            "distance_to_20d_low",
            "distance_to_60d_low",
            "volatility_5d",
            "volatility_10d",
            "volatility_60d",
            "downside_volatility_20d",
            "amplitude_5d",
            "amplitude_60d",
            "amount_avg_5d",
            "amount_avg_20d",
            "amount_avg_60d",
            "amount_zscore_20d",
            "amihud_illiquidity_20d",
            "limit_up_count_5d",
            "limit_up_count_10d",
            "limit_up_count_20d",
            "limit_down_count_5d",
            "limit_down_count_10d",
            "limit_down_count_20d",
            "close_position_in_range_20d",
            "main_net_amount_sum_3d",
            "main_net_amount_sum_5d",
            "main_net_amount_sum_10d",
            "main_net_to_amount_sum_3d",
            "main_net_to_amount_sum_5d",
            "main_net_to_amount_sum_10d",
            "main_net_to_circ_mv_sum_3d",
            "main_net_to_circ_mv_sum_5d",
            "main_net_to_circ_mv_sum_10d",
            "large_order_net_sum_3d",
            "large_order_net_sum_5d",
            "large_order_net_sum_10d",
            "extra_large_order_net_sum_3d",
            "extra_large_order_net_sum_5d",
            "extra_large_order_net_sum_10d",
            "moneyflow_positive_days_3d",
            "moneyflow_positive_days_5d",
            "moneyflow_positive_days_10d",
            "relative_industry_strength_5d",
            "relative_industry_strength_60d",
            "industry_rank_pct_5d",
            "industry_rank_pct_20d",
            "industry_rank_pct_60d",
            "market_return_1d",
            "market_return_5d",
            "market_return_20d",
            "market_volatility_20d",
            "market_drawdown_20d",
            "roe",
            "roe_dt",
            "roe_waa",
            "roa",
            "roic",
            "grossprofit_margin",
            "netprofit_margin",
            "ocf_to_profit",
            "debt_to_assets",
            "current_ratio",
            "quick_ratio",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "tr_yoy",
            "or_yoy",
            "ocf_yoy",
            "q_roe",
            "q_sales_yoy",
            "q_profit_yoy",
            "q_profit_qoq",
            "q_op_qoq",
            "q_gsprofit_margin",
            "q_netprofit_margin",
            "assets_yoy",
            "eqt_yoy",
            "rd_exp",
            "quality_composite",
            "growth_composite",
            "value_composite",
            "cross_return_5d_main_net_5d",
            "cross_industry_strength_flow_5d",
            "cross_quality_value",
        ]
        if factor_set == FACTOR_SET_V3:
            return sorted((set(v1) | set(v2_extra)) - FACTOR_SET_V3_EXCLUDED_FACTORS)
        v4_extra = {
            "short_vs_medium_return",
            "medium_vs_long_return",
            "volatility_change_5d_20d",
            "amount_trend_5d_20d",
            "range_pressure_20d",
            "limit_pressure_5d",
            "moneyflow_persistence_5d_10d",
            "industry_strength_risk_adjusted",
            "flow_risk_adjusted_5d",
            "flow_momentum_alignment_5d_20d",
            "value_risk_adjusted",
        }
        v4_factors = ((set(v1) | set(v2_extra)) - FACTOR_SET_V4_EXCLUDED_FACTORS) | v4_extra
        if factor_set == FACTOR_SET_V4:
            return sorted(v4_factors)
        if factor_set == FACTOR_SET_V5:
            v5_extra = {
                "cs_rank_return_5d",
                "cs_rank_return_20d",
                "cs_rank_relative_industry_strength_20d",
                "cs_rank_main_net_to_amount_sum_5d",
                "cs_rank_main_net_to_circ_mv_sum_5d",
                "cs_rank_volume_ratio",
                "cs_rank_turnover_rate",
                "cs_rank_volatility_20d",
                "cs_rank_amount_trend_5d_20d",
                "cs_rank_limit_pressure_5d",
                "market_breadth_return_5d",
                "market_breadth_flow_5d",
                "market_return_dispersion_5d",
                "relative_strength_momentum_alignment",
                "flow_persistence_intensity",
                "downside_protected_momentum_20d",
            }
            return sorted(v4_factors | v5_extra)
        return sorted(set(v1) | set(v2_extra))

    @staticmethod
    def _higher_is_better(factor_name: str) -> bool:
        low_is_good = {
            "circ_mv",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
            "volatility_60d",
            "downside_volatility_20d",
            "amplitude_5d",
            "amplitude_20d",
            "amplitude_60d",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "amihud_illiquidity_20d",
            "debt_to_assets",
            "limit_down_count_5d",
            "limit_down_count_10d",
            "limit_down_count_20d",
        }
        return factor_name not in low_is_good

    @staticmethod
    def _main_net_amount(row: dict[str, Any]) -> float | None:
        explicit = safe_float(row.get("net_mf_amount"))
        if explicit is not None:
            return explicit
        buy_lg = safe_float(row.get("buy_lg_amount")) or 0.0
        buy_elg = safe_float(row.get("buy_elg_amount")) or 0.0
        sell_lg = safe_float(row.get("sell_lg_amount")) or 0.0
        sell_elg = safe_float(row.get("sell_elg_amount")) or 0.0
        if all(row.get(key) is None for key in ["buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]):
            return None
        return buy_lg + buy_elg - sell_lg - sell_elg

    @staticmethod
    def _main_net_amount_series(df: pd.DataFrame) -> pd.Series:
        explicit = pd.to_numeric(df.get("net_mf_amount"), errors="coerce")
        components = df[["buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]].apply(pd.to_numeric, errors="coerce")
        derived = components[["buy_lg_amount", "buy_elg_amount"]].fillna(0).sum(axis=1) - components[
            ["sell_lg_amount", "sell_elg_amount"]
        ].fillna(0).sum(axis=1)
        all_missing = components.isna().all(axis=1)
        return explicit.where(explicit.notna(), derived.mask(all_missing))

    @staticmethod
    def _period_return(close: pd.Series, days: int) -> float | None:
        clean = close.dropna()
        if len(clean) <= days:
            return None
        start = clean.iloc[-days - 1]
        end = clean.iloc[-1]
        return safe_float(end / start - 1.0) if start else None

    @staticmethod
    def _distance_to_high(close: pd.Series, high: pd.Series, days: int) -> float | None:
        close_clean = close.dropna()
        high_clean = high.dropna().tail(days)
        if close_clean.empty or high_clean.empty:
            return None
        recent_high = high_clean.max()
        return safe_float(close_clean.iloc[-1] / recent_high - 1.0) if recent_high else None

    @staticmethod
    def _drawdown(close: pd.Series, high: pd.Series, days: int) -> float | None:
        return FactorLabService._distance_to_high(close, high, days)

    @staticmethod
    def _amplitude(high: pd.Series, low: pd.Series, close: pd.Series, days: int) -> float | None:
        high_tail = high.dropna().tail(days)
        low_tail = low.dropna().tail(days)
        close_tail = close.dropna().tail(days)
        if high_tail.empty or low_tail.empty or close_tail.empty:
            return None
        base = close_tail.mean()
        return safe_float((high_tail.max() - low_tail.min()) / base) if base else None

    @staticmethod
    def _listing_days(list_date: str | None, trade_date: str) -> int | None:
        if not list_date:
            return None
        try:
            return (datetime.strptime(trade_date, "%Y%m%d") - datetime.strptime(list_date, "%Y%m%d")).days
        except ValueError:
            return None

    @staticmethod
    def _listing_days_series(list_dates: pd.Series, trade_date: str) -> pd.Series:
        trade_ts = pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
        list_ts = pd.to_datetime(list_dates, format="%Y%m%d", errors="coerce")
        return (trade_ts - list_ts).dt.days

    @staticmethod
    def _market_cap_bucket(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        try:
            return pd.qcut(numeric.rank(method="first"), 3, labels=["small", "mid", "large"])
        except ValueError:
            return pd.Series(["unknown"] * len(values), index=values.index)

    @staticmethod
    def _market_regime(df: pd.DataFrame) -> str:
        pct = pd.to_numeric(df["pct_chg"], errors="coerce")
        breadth = (pct > 0).mean()
        avg_return = pct.mean()
        if breadth >= 0.60 and avg_return >= 0.5:
            return "green"
        if breadth <= 0.40 and avg_return <= -0.5:
            return "red"
        return "yellow"

    @staticmethod
    def _safe_quantile_labels(col: pd.Series, quantiles: int) -> pd.Series:
        clean_count = col.dropna().nunique()
        if clean_count < 2:
            return pd.Series([None] * len(col), index=col.index)
        bucket_count = min(quantiles, clean_count)
        try:
            return pd.qcut(col.rank(method="first"), bucket_count, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series([None] * len(col), index=col.index)


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "item"):
        return _json_clean(value.item())
    return value
