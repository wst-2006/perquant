"""Generate a read-only, reproducible historical data availability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.ml_lab.data_availability_audit import DataAvailabilityAuditor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local historical financial and event availability without changing data.")
    parser.add_argument("--output-dir", default="reports/ml_lab/data_availability_audit_v1")
    parser.add_argument("--artifact-path", default="reports/ml_lab/v11_dual_artifact_feature_v2/artifact.json")
    parser.add_argument("--checkpoints", default="20240401,20250603,20260601")
    args = parser.parse_args()
    result = DataAvailabilityAuditor(
        artifact_path=args.artifact_path,
        checkpoints=tuple(item.strip() for item in args.checkpoints.split(",") if item.strip()),
    ).run(args.output_dir)
    print(json.dumps({"json_path": result["json_path"], "markdown_path": result["markdown_path"], "conclusion": result["conclusion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
