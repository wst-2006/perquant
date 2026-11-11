from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.foundation_gate import FoundationGateConfig, FoundationGateError, assert_diagnostic_foundation_ready, assert_foundation_ready

DEFAULT_OUTPUT = Path("reports/ml_lab/gpu_experiment_package.zip")
DEFAULT_INCLUDE_PATHS = (
    "src",
    "scripts",
    "requirements-gpu.txt",
    "reports/ml_lab/training_frame_full.csv",
    "reports/ml_lab/experiment_manifest.json",
)
EXCLUDE_NAMES = {".env", ".env.example", "__pycache__", ".pytest_cache", ".venv", ".conda", ".idea"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
SECRET_HINTS = ("token", "password", "passwd", "secret", "db_password", "tushare_token")


def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    lower = path.as_posix().lower()
    if any(hint in lower for hint in SECRET_HINTS):
        return False
    return True


def iter_package_files(include_paths: tuple[str, ...] = DEFAULT_INCLUDE_PATHS) -> list[Path]:
    files: list[Path] = []
    for item in include_paths:
        item_path = Path(item)
        path = item_path if item_path.is_absolute() else PROJECT_ROOT / item_path
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if should_include(Path(archive_name(path))):
                files.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and should_include(Path(archive_name(child))):
                files.append(child)
    return sorted(set(files))


def archive_name(file: Path) -> str:
    try:
        return file.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return file.name


def assert_no_forbidden_files(files: list[Path]) -> None:
    forbidden = []
    for file in files:
        rel = archive_name(file)
        if rel == ".env" or any(hint in rel.lower() for hint in SECRET_HINTS):
            forbidden.append(rel)
    if forbidden:
        raise RuntimeError(f"Forbidden files would be packaged: {forbidden}")


def build_package(
    output_path: str | Path = DEFAULT_OUTPUT,
    training_frame_path: str | Path = "reports/ml_lab/training_frame_full.csv",
    manifest_path: str | Path = "reports/ml_lab/experiment_manifest.json",
    extra_paths: list[str | Path] | None = None,
    require_foundation_ready: bool = False,
    foundation_start_date: str = "20240401",
    foundation_end_date: str = "20260601",
    allow_blocked_foundation: bool = False,
    gate_mode: str = "production",
    gate_factor_set: str = "factor_lab_v2",
) -> dict:
    gate = None
    if require_foundation_ready:
        config = FoundationGateConfig(start_date=foundation_start_date, end_date=foundation_end_date)
        if gate_mode == "diagnostic":
            gate = assert_diagnostic_foundation_ready(config, factor_set=gate_factor_set)
        else:
            gate = assert_foundation_ready(config, allow_blocked=allow_blocked_foundation)
    output = PROJECT_ROOT / output_path
    if Path(output_path).is_absolute():
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    include_paths = (
        "src",
        "scripts",
        "requirements-gpu.txt",
        str(training_frame_path).replace("\\", "/"),
        str(manifest_path).replace("\\", "/"),
        *(str(path).replace("\\", "/") for path in (extra_paths or [])),
    )
    files = iter_package_files(include_paths)
    assert_no_forbidden_files(files)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            archive.write(file, archive_name(file))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_path": output.relative_to(PROJECT_ROOT).as_posix() if output.resolve().is_relative_to(PROJECT_ROOT) else output.name,
        "file_count": len(files),
        "size_bytes": output.stat().st_size,
        "included_roots": list(include_paths),
        "excluded": sorted(EXCLUDE_NAMES),
        "contains_env": False,
        "contains_database": False,
        "contains_secret_named_file": False,
    }
    if gate is not None:
        manifest["foundation_gate_status"] = gate["gate_status"]
        manifest["foundation_gate_blocking"] = gate["gate_blocking"]
        manifest["foundation_override_used"] = bool(gate.get("override_used"))
        manifest["gate_mode"] = gate.get("gate_mode", gate_mode)
        manifest["diagnostic_only"] = bool(gate.get("diagnostic_only") or gate.get("override_used"))
        manifest["promotion_allowed"] = bool(gate.get("promotion_allowed", gate["gate_status"] == "pass")) and not manifest["diagnostic_only"]
        manifest["promotion_policy"] = "GPU packages with diagnostic_only=true are research artifacts and must not be promoted."
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a safe offline package for GPU ML experiments.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    parser.add_argument("--manifest-path", default="reports/ml_lab/experiment_manifest.json")
    parser.add_argument("--extra-path", action="append", default=[])
    parser.add_argument("--foundation-start-date", default="20240401")
    parser.add_argument("--foundation-end-date", default="20260601")
    parser.add_argument("--gate-mode", choices=["production", "diagnostic"], default="production")
    parser.add_argument("--gate-factor-set", default="factor_lab_v2")
    parser.add_argument("--allow-blocked-foundation", action="store_true")
    args = parser.parse_args()
    result = build_package(
        args.output_path,
        training_frame_path=args.training_frame_path,
        manifest_path=args.manifest_path,
        extra_paths=args.extra_path,
        require_foundation_ready=True,
        foundation_start_date=args.foundation_start_date,
        foundation_end_date=args.foundation_end_date,
        allow_blocked_foundation=args.allow_blocked_foundation,
        gate_mode=args.gate_mode,
        gate_factor_set=args.gate_factor_set,
    )
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FoundationGateError as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": "foundation_gate_blocked",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2)
