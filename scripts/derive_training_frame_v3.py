from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ml_experiment_manifest import build_manifest
from src.quant_trading.ml_lab.service import MlLabService


PASSTHROUGH_COLUMNS = {
    "trade_date",
    "ts_code",
    "stable_score",
    "aggressive_score",
    "stable_rank",
    "aggressive_rank",
    "label_buy_date",
    "label_sell_date",
    "label_tradable",
    "label_net_return",
    "label_stable_positive",
    "label_top5",
    "label_top10",
}

DERIVED_RATIO_FEATURES = {
    "feature_main_net_to_amount_sum_3d": ("feature_main_net_amount_sum_3d", "feature_amount"),
    "feature_main_net_to_amount_sum_5d": ("feature_main_net_amount_sum_5d", "feature_amount"),
    "feature_main_net_to_amount_sum_10d": ("feature_main_net_amount_sum_10d", "feature_amount"),
    "feature_main_net_to_circ_mv_sum_3d": ("feature_main_net_amount_sum_3d", "feature_circ_mv"),
    "feature_main_net_to_circ_mv_sum_5d": ("feature_main_net_amount_sum_5d", "feature_circ_mv"),
    "feature_main_net_to_circ_mv_sum_10d": ("feature_main_net_amount_sum_10d", "feature_circ_mv"),
}

DERIVED_V4_FEATURES = {
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
}

DERIVED_V5_ROW_FEATURES = {
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

DERIVED_ROW_FEATURES = {**DERIVED_V4_FEATURES, **DERIVED_V5_ROW_FEATURES}


def _source_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _read_columns(source_columns: list[str], feature_columns: list[str]) -> list[str]:
    dependency_columns = {
        dep
        for col in feature_columns
        for dep in DERIVED_RATIO_FEATURES.get(col, ())
    }
    dependency_columns |= {
        dep
        for col in feature_columns
        for dep in DERIVED_ROW_FEATURES.get(col, ())[1:]
    }
    dependency_columns |= {source for col, source in V5_CROSS_SECTIONAL_RANK_SOURCES.items() if col in feature_columns}
    dependency_columns |= {spec[1] for col, spec in V5_MARKET_BREADTH_SOURCES.items() if col in feature_columns}
    requested = set(PASSTHROUGH_COLUMNS) | set(feature_columns) | dependency_columns
    return [col for col in source_columns if col in requested]


def _missing_non_derivable(source_columns: list[str], feature_columns: list[str]) -> list[str]:
    source_set = set(source_columns)
    missing = []
    for col in feature_columns:
        if col in source_set:
            continue
        deps = DERIVED_RATIO_FEATURES.get(col)
        if not deps or any(dep not in source_set for dep in deps):
            row_deps = DERIVED_ROW_FEATURES.get(col, ())[1:]
            group_source = V5_CROSS_SECTIONAL_RANK_SOURCES.get(col) or V5_MARKET_BREADTH_SOURCES.get(col, (None, None))[1]
            dependencies = row_deps or ((group_source,) if group_source else ())
            derivable = set(DERIVED_RATIO_FEATURES) | set(DERIVED_ROW_FEATURES)
            if not dependencies or any(dep not in source_set and dep not in derivable for dep in dependencies):
                missing.append(col)
    return sorted(missing)


def _apply_derived_features(chunk: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    derived_ratio_columns: dict[str, pd.Series] = {}
    for col in feature_columns:
        deps = DERIVED_RATIO_FEATURES.get(col)
        if not deps or col in chunk.columns:
            continue
        numerator = pd.to_numeric(chunk[deps[0]], errors="coerce")
        denominator = pd.to_numeric(chunk[deps[1]], errors="coerce")
        derived_ratio_columns[col] = numerator / denominator.where(denominator != 0)
    if derived_ratio_columns:
        chunk = pd.concat([chunk, pd.DataFrame(derived_ratio_columns, index=chunk.index)], axis=1)

    derived_row_columns: dict[str, pd.Series] = {}
    for col in feature_columns:
        spec = DERIVED_ROW_FEATURES.get(col)
        if not spec or col in chunk.columns:
            continue
        op, left_col, right_col = spec
        if left_col not in chunk.columns or right_col not in chunk.columns:
            continue
        left = pd.to_numeric(chunk[left_col], errors="coerce")
        right = pd.to_numeric(chunk[right_col], errors="coerce")
        if op == "sub":
            derived_row_columns[col] = left - right
        elif op == "add":
            derived_row_columns[col] = left + right
        elif op == "mul":
            derived_row_columns[col] = left * right
        elif op == "ratio":
            derived_row_columns[col] = left / right.where(right != 0)
        elif op == "ratio_minus_one":
            derived_row_columns[col] = left / right.where(right != 0) - 1.0
    if derived_row_columns:
        chunk = pd.concat([chunk, pd.DataFrame(derived_row_columns, index=chunk.index)], axis=1)
    return _apply_v5_cross_sectional_features(chunk, feature_columns)


def _apply_v5_cross_sectional_features(chunk: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if "trade_date" not in chunk.columns:
        return chunk
    dates = chunk["trade_date"].astype(str)
    derived: dict[str, pd.Series] = {}
    for target, source in V5_CROSS_SECTIONAL_RANK_SOURCES.items():
        if target in feature_columns and target not in chunk.columns and source in chunk.columns:
            derived[target] = pd.to_numeric(chunk[source], errors="coerce").groupby(dates, sort=False).rank(pct=True)
    for target, (operation, source) in V5_MARKET_BREADTH_SOURCES.items():
        if target not in feature_columns or target in chunk.columns or source not in chunk.columns:
            continue
        values = pd.to_numeric(chunk[source], errors="coerce")
        if operation == "positive_rate":
            derived[target] = values.gt(0).where(values.notna()).groupby(dates, sort=False).transform("mean")
        elif operation == "std":
            derived[target] = values.groupby(dates, sort=False).transform("std")
    if derived:
        chunk = pd.concat([chunk, pd.DataFrame(derived, index=chunk.index)], axis=1)
    return chunk


def _complete_trade_date_chunks(source: Path, read_columns: list[str], chunksize: int):
    carry = pd.DataFrame()
    previous_date = ""
    for chunk in pd.read_csv(source, usecols=read_columns, chunksize=chunksize):
        dates = chunk["trade_date"].astype(str)
        if not dates.is_monotonic_increasing or (previous_date and dates.iloc[0] < previous_date):
            raise ValueError("Training frame must be sorted by trade_date before deriving cross-sectional features.")
        previous_date = dates.iloc[-1]
        combined = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        combined_dates = combined["trade_date"].astype(str)
        final_date = combined_dates.iloc[-1]
        complete = combined[combined_dates != final_date]
        carry = combined[combined_dates == final_date].copy()
        if not complete.empty:
            yield complete
    if not carry.empty:
        yield carry


def _profile_features(
    source: Path,
    read_columns: list[str],
    feature_columns: list[str],
    chunksize: int,
) -> tuple[int, dict[str, int]]:
    rows = 0
    non_null_counts = {col: 0 for col in feature_columns}
    for chunk in _complete_trade_date_chunks(source, read_columns, chunksize):
        chunk = _apply_derived_features(chunk, feature_columns)
        rows += len(chunk)
        for col in feature_columns:
            values = pd.to_numeric(chunk[col], errors="coerce")
            non_null_counts[col] += int(values.notna().sum())
    return rows, non_null_counts


def _feature_gate_from_counts(rows: int, non_null_counts: dict[str, int]) -> dict[str, Any]:
    blocking = []
    dropped = []
    details = []
    for col, non_null in sorted(non_null_counts.items()):
        missing_rate = 1.0 if rows <= 0 else 1.0 - (non_null / rows)
        optional = MlLabService.is_optional_feature(col)
        threshold = 0.20 if optional else 0.05
        status = "pass" if missing_rate <= threshold else "dropped" if optional else "blocked"
        item = {
            "column": col,
            "missing_rate": round(float(missing_rate), 6),
            "non_null_rows": int(non_null),
            "required": not optional,
            "threshold": threshold,
            "status": status,
        }
        details.append(item)
        if status == "blocked":
            blocking.append(item)
        if status == "dropped":
            dropped.append(item)
    details.sort(key=lambda row: (row["status"] == "pass", -row["missing_rate"], row["column"]))
    return {
        "status": "pass" if not blocking else "blocked",
        "rows": rows,
        "feature_count": len(non_null_counts),
        "required_max_missing_rate": 0.05,
        "optional_max_missing_rate": 0.20,
        "blocking_count": len(blocking),
        "dropped_count": len(dropped),
        "blocking_features": blocking,
        "dropped_optional_features": dropped,
        "feature_missingness": details,
        "policy": "Required features block above 5% missingness; optional features above 20% are dropped before training.",
    }


def derive_frame(
    source_path: str | Path,
    output_path: str | Path,
    *,
    feature_set: str = "factor_lab_v3",
    chunksize: int = 200_000,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    feature_names = MlLabService.feature_names(feature_set)
    candidate_feature_columns = [f"feature_{name}" for name in feature_names]
    source_columns = _source_columns(source)
    missing_feature_columns = _missing_non_derivable(source_columns, candidate_feature_columns)
    if missing_feature_columns:
        raise ValueError(f"Source training frame is missing v3 feature columns: {missing_feature_columns}")

    read_columns = _read_columns(source_columns, candidate_feature_columns)
    rows, non_null_counts = _profile_features(source, read_columns, candidate_feature_columns, chunksize)
    candidate_gate = _feature_gate_from_counts(rows, non_null_counts)
    if candidate_gate["status"] != "pass":
        raise ValueError(f"Required v3 features failed missingness gate: {candidate_gate['blocking_features']}")

    dropped_columns = {row["column"] for row in candidate_gate["dropped_optional_features"]}
    active_feature_columns = [col for col in candidate_feature_columns if col not in dropped_columns]
    active_feature_names = [col.replace("feature_", "", 1) for col in active_feature_columns]
    active_gate = _feature_gate_from_counts(
        rows,
        {col: non_null_counts[col] for col in active_feature_columns},
    )

    first = True
    written_rows = 0
    passthrough_order = [col for col in sorted(PASSTHROUGH_COLUMNS) if col in source_columns]
    for chunk in _complete_trade_date_chunks(source, read_columns, chunksize):
        chunk = _apply_derived_features(chunk, active_feature_columns)
        ordered_columns = passthrough_order + active_feature_columns
        chunk = chunk[ordered_columns]
        chunk.to_csv(output, index=False, encoding="utf-8-sig", mode="w" if first else "a", header=first)
        first = False
        written_rows += len(chunk)

    feature_contract = MlLabService.build_feature_manifest(active_feature_names)
    result = {
        "task": "derive_training_frame_v3",
        "source_path": str(source),
        "output_path": str(output),
        "feature_set": feature_set,
        "rows": written_rows,
        "candidate_feature_count": len(candidate_feature_columns),
        "feature_count": len(active_feature_columns),
        "dropped_optional_features": candidate_gate["dropped_optional_features"],
        "feature_contract": feature_contract,
        "feature_missingness_gate": active_gate,
        "candidate_feature_missingness_gate": candidate_gate,
        "derivation_policy": "Derived from the source training frame by applying the requested real-data feature contract; no synthetic data is created.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a contract-clean training frame from an audited source training frame.")
    parser.add_argument("--source-path", default="reports/ml_lab/training_frame_v2_full.csv")
    parser.add_argument("--output-path", default="reports/ml_lab/training_frame_v3_full.csv")
    parser.add_argument("--feature-set", default="factor_lab_v3")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    result = derive_frame(args.source_path, args.output_path, feature_set=args.feature_set, chunksize=args.chunksize)
    manifest_path = Path(args.manifest_path or Path(args.output_path).with_suffix(".data_dependency_manifest.json"))
    manifest = build_manifest(
        args.output_path,
        {
            "factor_set": args.feature_set,
            "feature_set": args.feature_set,
            "source_training_frame": str(args.source_path),
            "derivation": "apply_requested_feature_contract",
            "report_contract_version": "ml_research_contract_v2",
        },
        notes="derived training frame; no secrets included; no synthetic data created",
        run_mode="production",
        feature_contract=result["feature_contract"],
    )
    manifest["source_training_frame"] = str(args.source_path)
    manifest["feature_missingness_gate"] = result["feature_missingness_gate"]
    manifest["candidate_feature_missingness_gate"] = result["candidate_feature_missingness_gate"]
    manifest["dropped_optional_features"] = result["dropped_optional_features"]
    manifest["derivation_policy"] = result["derivation_policy"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result["data_dependency_manifest_path"] = str(manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
