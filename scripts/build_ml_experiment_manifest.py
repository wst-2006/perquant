from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.data_sync import _asset_class_for_ts_code

DEFAULT_PARAMS = {
    "train_days": 120,
    "validation_days": 20,
    "step_days": 20,
    "top_n": [10, 20, 50],
    "models": ["lightgbm", "xgboost", "catboost"],
    "random_seed": 42,
}
SENSITIVE_COLUMN_HINTS = ("password", "token", "secret", "account", "账户", "资金", "password", "passwd", "pwd")


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_training_frame(path: Path, chunksize: int = 200_000) -> dict[str, Any]:
    reader = pd.read_csv(path, chunksize=chunksize)
    rows = 0
    columns: list[str] | None = None
    min_date: str | None = None
    max_date: str | None = None
    asset_class_counts: dict[str, int] = {}
    unique_codes: set[str] = set()
    for chunk in reader:
        if columns is None:
            columns = [str(col) for col in chunk.columns]
        rows += len(chunk)
        if "trade_date" in chunk.columns and not chunk.empty:
            dates = chunk["trade_date"].astype(str)
            chunk_min = dates.min()
            chunk_max = dates.max()
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)
        if "ts_code" in chunk.columns and not chunk.empty:
            for code in chunk["ts_code"].dropna().astype(str):
                unique_codes.add(code)
    columns = columns or []
    for code in unique_codes:
        asset_class = _asset_class_for_ts_code(code)
        asset_class_counts[asset_class] = asset_class_counts.get(asset_class, 0) + 1
    return {
        "training_frame_rows": rows,
        "training_frame_column_count": len(columns),
        "training_frame_columns": columns,
        "date_range": {"start": min_date, "end": max_date},
        "training_frame_unique_codes": len(unique_codes),
        "training_frame_asset_class_counts": asset_class_counts,
        "sensitive_column_warnings": sensitive_column_warnings(columns),
    }


def sensitive_column_warnings(columns: list[str]) -> list[str]:
    warnings = []
    for column in columns:
        lowered = column.lower()
        if any(hint in lowered for hint in SENSITIVE_COLUMN_HINTS):
            warnings.append(column)
    return warnings


def build_manifest(
    training_frame_path: str | Path,
    experiment_params: dict[str, Any] | None = None,
    notes: str = "no secrets included",
    *,
    run_mode: str = "production",
    foundation_gate: dict[str, Any] | None = None,
    data_dependencies: dict[str, Any] | None = None,
    training_frame_universe_audit: dict[str, Any] | None = None,
    feature_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(training_frame_path)
    if not path.exists():
        raise FileNotFoundError(path)
    profile = profile_training_frame(path)
    params = {**DEFAULT_PARAMS, **(experiment_params or {})}
    foundation_ok = foundation_gate is None or foundation_gate.get("gate_status") == "pass"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_frame_path": project_relative(path),
        "training_frame_sha256": sha256_file(path),
        "training_frame_size_bytes": path.stat().st_size,
        **profile,
        "python_version": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "random_seed": params.get("random_seed", 42),
        "experiment_params": params,
        "run_mode": run_mode,
        "diagnostic_only": run_mode == "diagnostic",
        "promotion_allowed": run_mode != "diagnostic" and foundation_ok and not bool((foundation_gate or {}).get("override_used")),
        "foundation_gate": foundation_gate,
        "data_dependencies": data_dependencies,
        "training_frame_universe_audit": training_frame_universe_audit,
        "feature_contract": feature_contract,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible manifest for ML GPU research experiments.")
    parser.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    parser.add_argument("--output-path", default="reports/ml_lab/experiment_manifest.json")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--validation-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--top-n", default="10,20,50")
    parser.add_argument("--models", default="lightgbm,xgboost,catboost")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--notes", default="no secrets included")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {
        "train_days": args.train_days,
        "validation_days": args.validation_days,
        "step_days": args.step_days,
        "top_n": [int(item.strip()) for item in args.top_n.split(",") if item.strip()],
        "models": [item.strip() for item in args.models.split(",") if item.strip()],
        "random_seed": args.random_seed,
    }
    manifest = build_manifest(args.training_frame_path, params, notes=args.notes)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "output_path": project_relative(output_path), "sensitive_column_warnings": manifest["sensitive_column_warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
