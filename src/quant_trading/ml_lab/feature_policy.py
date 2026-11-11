"""Shared training-feature contracts for ML research and data export."""

from __future__ import annotations


OPTIONAL_FEATURE_TERMS = (
    "risk_adjusted",
    "flow_momentum_alignment",
    "flow_persistence",
    "moneyflow_persistence",
    "market_return",
    "medium_vs_long_return",
    "industry_rank_pct_60d",
    "ma_slope_60d",
    "relative_industry_strength_60d",
    "main_net",
    "large_order",
    "extra_large_order",
    "moneyflow",
    "cross_return_5d_main_net_5d",
    "cross_industry_strength_flow_5d",
    "cross_quality_value",
    "quality_composite",
    "growth_composite",
    "value_composite",
    "roe",
    "roa",
    "roic",
    "grossprofit_margin",
    "netprofit_margin",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "tr_yoy",
    "or_yoy",
    "ocf",
    "debt",
    "current_ratio",
    "quick_ratio",
    "assets_yoy",
    "eqt_yoy",
    "rd_exp",
    "q_",
    "return_120d",
    "distance_to_120d_high",
)


def is_optional_feature(name_or_column: str) -> bool:
    """Return whether a feature has the relaxed missingness contract."""
    name = name_or_column.replace("feature_", "", 1)
    return any(term in name for term in OPTIONAL_FEATURE_TERMS)
