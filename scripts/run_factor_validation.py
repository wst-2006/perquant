import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.factor_lab.batch import FactorLabBatchRunner, VALIDATION_FACTORS
from src.quant_trading.db import init_db


def parse_horizons(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first Factor Lab historical validation suite.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("readiness")
    sub.add_parser("build-snapshots")
    forward = sub.add_parser("build-forward-returns")
    forward.add_argument("--horizons", default="1,3,5,10,20")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--factors", default=",".join(VALIDATION_FACTORS))
    evaluate.add_argument("--horizons", default="1,3,5,10,20")
    evaluate.add_argument("--output-dir", default="reports/factor_lab")
    walk_forward = sub.add_parser("walk-forward")
    walk_forward.add_argument("--factor", required=True)
    walk_forward.add_argument("--train-window", type=int, default=60)
    walk_forward.add_argument("--validation-window", type=int, default=20)
    walk_forward.add_argument("--step-window", type=int, default=20)
    walk_forward.add_argument("--start-date", default="20250201")
    walk_forward.add_argument("--end-date", default="20260618")
    walk_forward.add_argument("--horizon", type=int, default=5)
    walk_forward.add_argument("--output-dir", default="reports/factor_lab")
    args = parser.parse_args()

    runner = FactorLabBatchRunner.create()
    if args.command == "readiness":
        result = runner.data_readiness()
    elif args.command == "build-snapshots":
        result = runner.build_snapshots_for_windows()
    elif args.command == "build-forward-returns":
        result = runner.build_forward_returns_for_windows(horizons=parse_horizons(args.horizons))
    elif args.command == "evaluate":
        factors = [item.strip() for item in args.factors.split(",") if item.strip()]
        init_db()
        result = runner.evaluate_validation_suite(
            factors=factors,
            horizons=parse_horizons(args.horizons),
            output_dir=args.output_dir,
        )
    elif args.command == "walk-forward":
        init_db()
        result = runner.walk_forward_validation(
            factor_name=args.factor,
            train_window=args.train_window,
            validation_window=args.validation_window,
            step_window=args.step_window,
            start_date=args.start_date,
            end_date=args.end_date,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
