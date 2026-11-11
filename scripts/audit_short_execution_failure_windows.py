"""Run the frozen final-OOS short-execution failure-window audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.ml_lab.failure_window_audit import ShortExecutionFailureWindowAuditor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-path", default="reports/ml_lab/short_execution_window_study_cpu_v3/short_execution_oos_top5.csv")
    parser.add_argument("--label-dir", default="reports/ml_lab/short_execution_labels_v1")
    parser.add_argument("--feature-dir", default="reports/ml_lab/training_frame_v9_events")
    parser.add_argument("--output-dir", default="reports/ml_lab/short_execution_failure_window_audit_v1")
    parser.add_argument("--raw-state-factor-set", default="factor_lab_v2")
    args = parser.parse_args()
    result = ShortExecutionFailureWindowAuditor(raw_state_factor_set=args.raw_state_factor_set).run(
        Path(args.selection_path), Path(args.label_dir), Path(args.feature_dir), Path(args.output_dir)
    )
    print(result["report_path"])


if __name__ == "__main__":
    main()
