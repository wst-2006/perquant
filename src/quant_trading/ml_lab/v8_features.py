"""Shared v5-compatible feature derivation for v8 offline and live scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .service import MlLabConfig, MlLabService
from .v8_events import AnnouncementEventStore, V8_EVENT_FEATURES
from .v9_events import V9_EVENT_FEATURES, V9EventStore


V8_LIVE_FEATURE_CONTRACT = "ml_v8_v5_feature_derivation_v2"


DERIVED_RATIO_FEATURES = {
    "feature_main_net_to_amount_sum_3d": ("feature_main_net_amount_sum_3d", "feature_amount"),
    "feature_main_net_to_amount_sum_5d": ("feature_main_net_amount_sum_5d", "feature_amount"),
    "feature_main_net_to_amount_sum_10d": ("feature_main_net_amount_sum_10d", "feature_amount"),
    "feature_main_net_to_circ_mv_sum_3d": ("feature_main_net_amount_sum_3d", "feature_circ_mv"),
    "feature_main_net_to_circ_mv_sum_5d": ("feature_main_net_amount_sum_5d", "feature_circ_mv"),
    "feature_main_net_to_circ_mv_sum_10d": ("feature_main_net_amount_sum_10d", "feature_circ_mv"),
}

DERIVED_ROW_FEATURES = {
    "feature_short_vs_medium_return": ("sub", "feature_return_5d", "feature_return_20d"),
    "feature_medium_vs_long_return": ("sub", "feature_return_20d", "feature_return_60d"),
    "feature_volatility_change_5d_20d": ("ratio_minus_one", "feature_volatility_5d", "feature_volatility_20d"),
    "feature_amount_trend_5d_20d": ("ratio_minus_one", "feature_amount_avg_5d", "feature_amount_avg_20d"),
    "feature_range_pressure_20d": ("add", "feature_distance_to_20d_high", "feature_distance_to_20d_low"),
    "feature_limit_pressure_5d": ("sub", "feature_limit_up_count_5d", "feature_limit_down_count_5d"),
    "feature_moneyflow_persistence_5d_10d": ("ratio", "feature_moneyflow_positive_days_5d", "feature_moneyflow_positive_days_10d"),
    "feature_industry_strength_risk_adjusted": ("ratio", "feature_relative_industry_strength_20d", "feature_volatility_20d"),
    "feature_flow_risk_adjusted_5d": ("ratio", "feature_main_net_to_amount_sum_5d", "feature_volatility_20d"),
    "feature_flow_momentum_alignment_5d_20d": ("mul", "feature_main_net_to_circ_mv_sum_5d", "feature_return_20d"),
    "feature_value_risk_adjusted": ("ratio", "feature_value_composite", "feature_volatility_20d"),
    "feature_relative_strength_momentum_alignment": ("mul", "feature_relative_industry_strength_20d", "feature_relative_industry_strength_5d"),
    "feature_flow_persistence_intensity": ("mul", "feature_main_net_to_amount_sum_5d", "feature_moneyflow_persistence_5d_10d"),
    "feature_downside_protected_momentum_20d": ("ratio", "feature_return_20d", "feature_downside_volatility_20d"),
}

V5_CROSS_SECTIONAL_RANK_SOURCES = {
    "feature_cs_rank_return_5d": "feature_return_5d",
    "feature_cs_rank_return_20d": "feature_return_20d",
    "feature_cs_rank_relative_industry_strength_20d": "feature_relative_industry_strength_20d",
    "feature_cs_rank_main_net_to_amount_sum_5d": "feature_main_net_to_amount_sum_5d",
    "feature_cs_rank_main_net_to_circ_mv_sum_5d": "feature_main_net_to_circ_mv_sum_5d",
    "feature_cs_rank_volume_ratio": "feature_volume_ratio",
    "feature_cs_rank_turnover_rate": "feature_turnover_rate",
    "feature_cs_rank_volatility_20d": "feature_volatility_20d",
    "feature_cs_rank_amount_trend_5d_20d": "feature_amount_trend_5d_20d",
    "feature_cs_rank_limit_pressure_5d": "feature_limit_pressure_5d",
}

V5_MARKET_BREADTH_SOURCES = {
    "feature_market_breadth_return_5d": ("positive_rate", "feature_return_5d"),
    "feature_market_breadth_flow_5d": ("positive_rate", "feature_main_net_to_amount_sum_5d"),
    "feature_market_return_dispersion_5d": ("std", "feature_return_5d"),
}


def derive_v5_features(frame: pd.DataFrame, requested_columns: list[str]) -> pd.DataFrame:
    # This contract derives every listed feature from its base inputs. Snapshots can
    # contain a same-named precomputed factor on a different normalization scale.
    result = frame.copy()
    needed = set(requested_columns)
    needed.update(source for target, source in V5_CROSS_SECTIONAL_RANK_SOURCES.items() if target in needed)
    needed.update(source for target, (_, source) in V5_MARKET_BREADTH_SOURCES.items() if target in needed)
    for _ in range(len(DERIVED_ROW_FEATURES) + len(DERIVED_RATIO_FEATURES)):
        before = len(needed)
        for target, dependencies in DERIVED_RATIO_FEATURES.items():
            if target in needed:
                needed.update(dependencies)
        for target, (_, left, right) in DERIVED_ROW_FEATURES.items():
            if target in needed:
                needed.update((left, right))
        if len(needed) == before:
            break
    additions: dict[str, pd.Series] = {}
    for target, dependencies in DERIVED_RATIO_FEATURES.items():
        if target not in needed:
            continue
        left, right = dependencies
        if left in result.columns and right in result.columns:
            numerator = pd.to_numeric(result[left], errors="coerce")
            denominator = pd.to_numeric(result[right], errors="coerce")
            derived = numerator / denominator.where(denominator != 0)
            if target in result.columns:
                result[target] = derived
            else:
                additions[target] = derived
    if additions:
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)

    for _ in range(len(DERIVED_ROW_FEATURES)):
        additions = {}
        for target, spec in DERIVED_ROW_FEATURES.items():
            if target not in needed:
                continue
            operation, left_name, right_name = spec
            if left_name not in result.columns or right_name not in result.columns:
                continue
            left = pd.to_numeric(result[left_name], errors="coerce")
            right = pd.to_numeric(result[right_name], errors="coerce")
            if operation == "sub":
                derived = left - right
            elif operation == "add":
                derived = left + right
            elif operation == "mul":
                derived = left * right
            elif operation == "ratio":
                derived = left / right.where(right != 0)
            elif operation == "ratio_minus_one":
                derived = left / right.where(right != 0) - 1.0
            else:
                continue
            if target in result.columns:
                result[target] = derived
            else:
                additions[target] = derived
        if not additions:
            break
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)

    dates = result["trade_date"].astype(str)
    additions = {}
    for target, source in V5_CROSS_SECTIONAL_RANK_SOURCES.items():
        if target in requested_columns and source in result.columns:
            derived = pd.to_numeric(result[source], errors="coerce").groupby(dates, sort=False).rank(pct=True)
            if target in result.columns:
                result[target] = derived
            else:
                additions[target] = derived
    for target, (operation, source) in V5_MARKET_BREADTH_SOURCES.items():
        if target not in requested_columns or source not in result.columns:
            continue
        values = pd.to_numeric(result[source], errors="coerce")
        if operation == "positive_rate":
            derived = values.gt(0).where(values.notna()).groupby(dates, sort=False).transform("mean")
        elif operation == "std":
            derived = values.groupby(dates, sort=False).transform("std")
        else:
            continue
        if target in result.columns:
            result[target] = derived
        else:
            additions[target] = derived
    if additions:
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
    return result


def build_live_v8_feature_frame(
    trade_date: str,
    feature_columns: list[str],
    *,
    announcement_store: AnnouncementEventStore | None = None,
    v9_event_store: V9EventStore | None = None,
) -> pd.DataFrame:
    """Build one full-market frame, optionally reusing frozen event stores across dates."""
    service = MlLabService(MlLabConfig(factor_set="factor_lab_v2", feature_set="factor_lab_v2"))
    source = service._score_frame(trade_date)
    if source.empty:
        raise ValueError(f"No factor_lab_v2 snapshots found for trade_date={trade_date}")
    ranked = service._rank_scores(source)
    records = []
    for row in ranked.to_dict("records"):
        normalized = row.get("features") or {}
        records.append(
            {
                "trade_date": str(trade_date),
                "ts_code": row["ts_code"],
                "name": row.get("name"),
                "industry": row.get("industry"),
                "stable_score": row.get("stable_score"),
                "aggressive_score": row.get("aggressive_score"),
                "stable_rank": row.get("stable_rank"),
                "aggressive_rank": row.get("aggressive_rank"),
                "risk_tags": service._risk_tags(row.get("raw_features") or {}, row.get("data_quality") or {}),
                **{f"feature_{name}": value for name, value in normalized.items()},
            }
        )
    frame = derive_v5_features(pd.DataFrame(records), feature_columns)
    event_columns = [name for name in feature_columns if name in V8_EVENT_FEATURES]
    if event_columns:
        event_features = (announcement_store or AnnouncementEventStore.from_database()).features_for_rows(
            frame[["trade_date", "ts_code"]]
        )
        frame = pd.concat([frame.reset_index(drop=True), event_features[event_columns].reset_index(drop=True)], axis=1)
    v9_event_columns = [name for name in feature_columns if name in V9_EVENT_FEATURES]
    if v9_event_columns:
        v9_event_features = (v9_event_store or V9EventStore.from_database()).features_for_rows(
            frame[["trade_date", "ts_code"]]
        )
        frame = pd.concat([frame.reset_index(drop=True), v9_event_features[v9_event_columns].reset_index(drop=True)], axis=1)
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Live v8 feature contract cannot derive columns: {missing}")
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    all_missing = [column for column in feature_columns if values[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"Live v8 feature columns are entirely missing: {all_missing}")
    frame[feature_columns] = values.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return frame
