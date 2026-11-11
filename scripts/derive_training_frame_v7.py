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
from src.quant_trading.ml_lab.market_state import MarketStateResearch
from src.quant_trading.ml_lab.research import MARKET_STATE_FEATURES


RAW_STATE_COLUMNS = [feature.replace("feature_", "feature_state_", 1) for feature in MARKET_STATE_FEATURES]


def _source_dates(source: Path, chunksize: int) -> list[str]:
    dates: set[str] = set()
    for chunk in pd.read_csv(source, usecols=["trade_date"], chunksize=chunksize):
        dates.update(chunk["trade_date"].astype(str))
    return sorted(dates)


def _raw_state_by_date(dates: list[str], raw_state_factor_set: str) -> tuple[pd.DataFrame, dict]:
    seed = pd.DataFrame({"trade_date": dates})
    state, provenance = MarketStateResearch.attach_raw_market_state(seed, raw_state_factor_set)
    renamed = state.rename(columns=dict(zip(MARKET_STATE_FEATURES, RAW_STATE_COLUMNS, strict=True)))
    return renamed[["trade_date", *RAW_STATE_COLUMNS]], provenance


def derive_frame(
    source_path: str | Path,
    output_path: str | Path,
    *,
    raw_state_factor_set: str = "factor_lab_v2",
    chunksize: int = 200_000,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in MARKET_STATE_FEATURES if column not in columns]
    if missing:
        raise ValueError(f"v5 source frame is missing normalized market columns: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state, provenance = _raw_state_by_date(_source_dates(source, chunksize), raw_state_factor_set)
    keep_columns = [column for column in columns if column not in MARKET_STATE_FEATURES]
    first = True
    rows = 0
    state_non_null = {column: 0 for column in RAW_STATE_COLUMNS}
    for chunk in pd.read_csv(source, chunksize=chunksize):
        chunk["trade_date"] = chunk["trade_date"].astype(str)
        derived = chunk[keep_columns].merge(state, on="trade_date", how="left", validate="many_to_one")
        for column in RAW_STATE_COLUMNS:
            state_non_null[column] += int(derived[column].notna().sum())
        derived.to_csv(output, index=False, encoding="utf-8-sig", mode="w" if first else "a", header=first)
        first = False
        rows += len(derived)
    coverage = {column: round(count / rows, 6) if rows else 0.0 for column, count in state_non_null.items()}
    if any(value < 0.95 for value in coverage.values()):
        raise ValueError(f"Raw state columns failed coverage gate: {coverage}")
    return {
        "task": "derive_training_frame_v7",
        "source_path": str(source),
        "output_path": str(output),
        "rows": rows,
        "removed_normalized_macro_columns": list(MARKET_STATE_FEATURES),
        "added_raw_state_columns": list(RAW_STATE_COLUMNS),
        "raw_state_factor_set": raw_state_factor_set,
        "raw_state_provenance": provenance,
        "raw_state_coverage": coverage,
        "derivation_policy": "Replace cross-section-normalized market-state columns with real Tushare-backed raw signal-date state features; no synthetic values are created.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive v7 training data with raw market-state features.")
    parser.add_argument("--source-path", default="reports/ml_lab/training_frame_v5_full.csv")
    parser.add_argument("--output-path", default="reports/ml_lab/training_frame_v7_full.csv")
    parser.add_argument("--raw-state-factor-set", default="factor_lab_v2")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--manifest-path", default=None)
    args = parser.parse_args()
    result = derive_frame(args.source_path, args.output_path, raw_state_factor_set=args.raw_state_factor_set, chunksize=args.chunksize)
    manifest = build_manifest(
        args.output_path,
        {"factor_set": "factor_lab_v7", "feature_set": "factor_lab_v7", "source_training_frame": str(args.source_path), "raw_state_factor_set": args.raw_state_factor_set},
        notes="v7 raw market-state training frame; no synthetic data created",
        run_mode="diagnostic",
    )
    manifest.update({key: result[key] for key in ("removed_normalized_macro_columns", "added_raw_state_columns", "raw_state_factor_set", "raw_state_provenance", "raw_state_coverage", "derivation_policy")})
    path = Path(args.manifest_path or Path(args.output_path).with_suffix(".data_dependency_manifest.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result["data_dependency_manifest_path"] = str(path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
