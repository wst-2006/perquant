"""Build and evaluate the frozen short-execution blind period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.ml_lab.short_execution_blind import ShortExecutionBlindRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-feature-dir", default="reports/ml_lab/training_frame_v9_events")
    parser.add_argument("--historical-label-dir", default="reports/ml_lab/short_execution_labels_v1")
    parser.add_argument("--event-snapshot-dir", default="reports/ml_lab/v9_event_snapshots_blind_20260602_20260709_v2")
    parser.add_argument("--output-dir", default="reports/ml_lab/short_execution_blind_20260602_20260709_v3")
    parser.add_argument("--train-days", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=120)
    args = parser.parse_args()
    result = ShortExecutionBlindRunner().run(
        historical_feature_dir=args.historical_feature_dir,
        historical_label_dir=args.historical_label_dir,
        event_snapshot_dir=args.event_snapshot_dir,
        output_dir=args.output_dir,
        train_days=args.train_days,
        n_estimators=args.n_estimators,
    )
    print(json.dumps({"report_path": result["report_path"], "status": result["status"], "blind_evaluation": result["blind_evaluation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
