import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.selection import SelectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a saved selection run to CSV.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = SelectionService.create().export_run_results_csv(args.run_id, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
