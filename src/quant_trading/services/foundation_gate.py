from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.tushare_client import FINA_INDICATOR_GATE_FIELDS
from .data_sync import DataSyncService


class FoundationGateError(RuntimeError):
    """Raised when research or training would run on an incomplete data foundation."""


@dataclass(frozen=True)
class FoundationGateConfig:
    start_date: str = "20240401"
    end_date: str = "20260601"
    min_coverage: float = 0.95
    required_coverage: tuple[str, ...] = (
        "daily_quotes",
        "daily_basic",
        "moneyflow",
        "index_daily",
        "stock_limit",
        "adj_factor",
    )
    require_financial_raw: bool = True


@dataclass(frozen=True)
class DataDependencyProfile:
    factor_set: str
    required_coverage: tuple[str, ...]
    required_financial_raw: tuple[str, ...]
    policy: str
    required_financial_fields: tuple[str, ...] = ()


FEATURE_DEPENDENCY_PROFILES = {
    "factor_lab_v1": DataDependencyProfile(
        factor_set="factor_lab_v1",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=(),
        policy="factor_lab_v1 uses market/quote/basic/moneyflow-derived features; raw financial endpoints are not feature dependencies.",
    ),
    "factor_lab_v2": DataDependencyProfile(
        factor_set="factor_lab_v2",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=("fina_indicator",),
        required_financial_fields=FINA_INDICATOR_GATE_FIELDS,
        policy=(
            "factor_lab_v2 uses fina_indicator plus quote/basic/moneyflow/index/limit/adjustment data. "
            "fina_audit and fina_mainbz remain production-foundation requirements, but are not current feature dependencies."
        ),
    ),
    "factor_lab_v2_financial_v2": DataDependencyProfile(
        factor_set="factor_lab_v2_financial_v2",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=("fina_indicator",),
        required_financial_fields=FINA_INDICATOR_GATE_FIELDS,
        policy=(
            "factor_lab_v2_financial_v2 preserves the v2 factor contract while selecting the newest eligible "
            "report period and the richest audited Tushare financial revision. Frozen factor_lab_v2 snapshots remain unchanged."
        ),
    ),
    "factor_lab_v3": DataDependencyProfile(
        factor_set="factor_lab_v3",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=("fina_indicator",),
        required_financial_fields=(
            "roe",
            "grossprofit_margin",
            "netprofit_margin",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "or_yoy",
        ),
        policy=(
            "factor_lab_v3 keeps only real Tushare-backed, announcement-date-safe features that pass missingness checks. "
            "Known 100%-missing v2 fields are excluded before training."
        ),
    ),
    "factor_lab_v4": DataDependencyProfile(
        factor_set="factor_lab_v4",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=("fina_indicator",),
        policy=(
            "factor_lab_v4 uses real Tushare-backed v3-clean features plus derived interaction/meta factors; "
            "all production/diagnostic runs remain blocked by the same universe, label, split, and missingness gates."
        ),
    ),
    "factor_lab_v5": DataDependencyProfile(
        factor_set="factor_lab_v5",
        required_coverage=(
            "daily_quotes",
            "daily_basic",
            "moneyflow",
            "index_daily",
            "stock_limit",
            "adj_factor",
        ),
        required_financial_raw=("fina_indicator",),
        policy=(
            "factor_lab_v5 retains the v4 real-data contract and adds same-day full-market ranks, market breadth, "
            "and risk-adjusted interactions. Every v5 input is available at the signal date; no future labels enter features."
        ),
    ),
    "factor_lab_v7": DataDependencyProfile(
        factor_set="factor_lab_v7",
        required_coverage=("daily_quotes", "daily_basic", "moneyflow", "index_daily", "stock_limit", "adj_factor"),
        required_financial_raw=("fina_indicator",),
        policy="factor_lab_v7 retains v5 stock features and replaces normalized market-state columns with raw, signal-date Tushare state features aggregated across the A-share market.",
    ),
    "factor_lab_v8": DataDependencyProfile(
        factor_set="factor_lab_v8",
        required_coverage=("daily_quotes", "daily_basic", "moneyflow", "index_daily", "stock_limit", "adj_factor"),
        required_financial_raw=("fina_indicator",),
        policy=(
            "factor_lab_v8 rebuilds the audited v5 stock feature contract from factor_lab_v2 snapshots, "
            "uses stock-specific real limit prices for multi-horizon labels, and isolates raw market-state features in the eligibility gate."
        ),
    ),
    "factor_lab_v8_events": DataDependencyProfile(
        factor_set="factor_lab_v8_events",
        required_coverage=("daily_quotes", "daily_basic", "moneyflow", "index_daily", "stock_limit", "adj_factor"),
        required_financial_raw=("fina_indicator",),
        policy=(
            "factor_lab_v8_events retains the audited v8 contract and adds sparse forecast, express, and dividend records. "
            "Sparse event absence remains missing and same-day announcements are delayed to the next signal date."
        ),
    ),
    "factor_lab_v9_events": DataDependencyProfile(
        factor_set="factor_lab_v9_events",
        required_coverage=("daily_quotes", "daily_basic", "moneyflow", "index_daily", "stock_limit", "adj_factor"),
        required_financial_raw=("fina_indicator",),
        policy=(
            "factor_lab_v9_events retains the audited v8 Top5 contract and adds Tushare-only dragon-tiger, limit-structure, "
            "suspension, repurchase, holder-trade, and unlock-schedule features. Trade-date events are delayed to the next signal date; "
            "announcement events require ann_date strictly before the signal date; raw non-A-share rows are filtered out of feature generation."
        ),
    ),
}


def dependency_profile_for_factor_set(factor_set: str) -> DataDependencyProfile | None:
    return FEATURE_DEPENDENCY_PROFILES.get(factor_set)


def inspect_foundation(config: FoundationGateConfig | None = None) -> dict[str, Any]:
    cfg = config or FoundationGateConfig()
    data_sync = DataSyncService(client=None)
    status = data_sync.foundation_status(cfg.start_date, cfg.end_date)
    stock_audit = data_sync.stock_universe_audit(preview_limit=5)
    coverage = status.get("coverage") or {}
    blocking = list(status.get("blocking") or [])

    for dataset in cfg.required_coverage:
        ratio = coverage.get(dataset)
        if ratio is None or ratio < cfg.min_coverage:
            item = f"{dataset}_coverage_below_{cfg.min_coverage:.2f}"
            if item not in blocking:
                blocking.append(item)

    if cfg.require_financial_raw and status.get("missing_financial_endpoints"):
        if "financial_raw_records" not in blocking:
            blocking.append("financial_raw_records")
    if stock_audit.get("status") != "pass":
        if "stock_universe_audit" not in blocking:
            blocking.append("stock_universe_audit")

    gate_status = "pass" if not blocking else "blocked"
    return {
        **status,
        "stock_universe_audit": stock_audit,
        "gate_mode": "production",
        "diagnostic_only": False,
        "promotion_allowed": gate_status == "pass",
        "gate_status": gate_status,
        "gate_blocking": blocking,
        "gate_policy": (
            "Training-frame export, model training, research-harness training, and GPU packaging "
            "must be blocked until this gate passes, unless explicitly overridden for diagnostics."
        ),
        "gate_config": {
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "min_coverage": cfg.min_coverage,
            "required_coverage": list(cfg.required_coverage),
            "require_financial_raw": cfg.require_financial_raw,
        },
    }


def inspect_diagnostic_foundation(
    config: FoundationGateConfig | None = None,
    *,
    factor_set: str = "factor_lab_v2",
) -> dict[str, Any]:
    cfg = config or FoundationGateConfig()
    data_sync = DataSyncService(client=None)
    status = data_sync.foundation_status(cfg.start_date, cfg.end_date)
    stock_audit = data_sync.stock_universe_audit(preview_limit=5)
    production_gate = inspect_foundation(cfg)
    profile = dependency_profile_for_factor_set(factor_set)

    blocking: list[str] = []
    dependency_details: dict[str, Any] = {
        "factor_set": factor_set,
        "known_profile": profile is not None,
    }
    if profile is None:
        blocking.append("unknown_factor_set_dependencies")
    else:
        coverage = status.get("coverage") or {}
        for dataset in profile.required_coverage:
            ratio = coverage.get(dataset)
            if ratio is None or ratio < cfg.min_coverage:
                blocking.append(f"{dataset}_coverage_below_{cfg.min_coverage:.2f}")

        min_raw_coverage = status.get("min_raw_stock_coverage") or 0.90
        raw_endpoint_rows = status.get("raw_endpoint_rows") or {}
        raw_endpoint_stock_coverage = status.get("raw_endpoint_stock_coverage") or {}
        missing_required_raw = []
        low_required_raw = []
        for endpoint in profile.required_financial_raw:
            if raw_endpoint_rows.get(endpoint, 0) <= 0:
                missing_required_raw.append(endpoint)
                continue
            ratio = raw_endpoint_stock_coverage.get(endpoint)
            if ratio is None or ratio < min_raw_coverage:
                low_required_raw.append(endpoint)
        if missing_required_raw:
            blocking.append("missing_required_feature_raw_records")
        if low_required_raw:
            blocking.append("required_feature_raw_coverage_below_threshold")
        min_field_coverage = status.get("min_financial_field_stock_coverage") or 0.90
        field_coverage = status.get("financial_field_stock_coverage") or {}
        low_required_fields = [
            field
            for field in profile.required_financial_fields
            if field_coverage.get(field) is None or field_coverage[field] < min_field_coverage
        ]
        if low_required_fields:
            blocking.append("required_financial_field_coverage_below_threshold")
        dependency_details.update(
            {
                "required_coverage": list(profile.required_coverage),
                "required_financial_raw": list(profile.required_financial_raw),
                "missing_required_financial_raw": missing_required_raw,
                "low_required_financial_raw": low_required_raw,
                "min_required_financial_raw_coverage": min_raw_coverage,
                "required_financial_fields": list(profile.required_financial_fields),
                "low_required_financial_fields": low_required_fields,
                "min_required_financial_field_coverage": min_field_coverage,
                "policy": profile.policy,
            }
        )

    if stock_audit.get("status") != "pass":
        blocking.append("stock_universe_audit")

    gate_status = "pass" if not blocking else "blocked"
    return {
        **status,
        "stock_universe_audit": stock_audit,
        "production_gate_status": production_gate.get("gate_status"),
        "production_gate_blocking": production_gate.get("gate_blocking", []),
        "gate_mode": "diagnostic",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "gate_status": gate_status,
        "gate_blocking": blocking,
        "data_dependency_profile": dependency_details,
        "gate_policy": (
            "Diagnostic runs may proceed only when the current feature dependencies pass. "
            "Outputs are diagnostic_only and must not be promoted while the production gate is blocked."
        ),
        "gate_config": {
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "min_coverage": cfg.min_coverage,
            "factor_set": factor_set,
        },
    }


def assert_foundation_ready(
    config: FoundationGateConfig | None = None,
    *,
    allow_blocked: bool = False,
) -> dict[str, Any]:
    result = inspect_foundation(config)
    if result["gate_status"] == "blocked" and not allow_blocked:
        blocking = ", ".join(result["gate_blocking"])
        raise FoundationGateError(
            f"Data foundation gate is blocked for {result['gate_config']['start_date']}-"
            f"{result['gate_config']['end_date']}: {blocking}. "
            "Run scripts/sync_data.py foundation-status for details, or pass the explicit override only for diagnostics."
        )
    if allow_blocked and result["gate_status"] == "blocked":
        return {
            **result,
            "override_used": True,
            "override_warning": "Blocked foundation was explicitly overridden; outputs must not be promoted.",
        }
    return result


def assert_diagnostic_foundation_ready(
    config: FoundationGateConfig | None = None,
    *,
    factor_set: str = "factor_lab_v2",
) -> dict[str, Any]:
    result = inspect_diagnostic_foundation(config, factor_set=factor_set)
    if result["gate_status"] == "blocked":
        blocking = ", ".join(result["gate_blocking"])
        raise FoundationGateError(
            f"Diagnostic data dependency gate is blocked for {result['gate_config']['start_date']}-"
            f"{result['gate_config']['end_date']} factor_set={factor_set}: {blocking}. "
            "Run scripts/sync_data.py foundation-status and stock-universe-audit for details."
        )
    return result
