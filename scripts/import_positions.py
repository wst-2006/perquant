import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.portfolio import PortfolioService


def main() -> None:
    parser = argparse.ArgumentParser(description="Import real account positions from CSV.")
    parser.add_argument("--csv", required=True, help="CSV path with ts_code,name,cost,quantity columns")
    parser.add_argument("--clear", action="store_true", help="Clear current holding positions before import")
    args = parser.parse_args()

    result = PortfolioService.create().import_positions_csv(args.csv, clear_existing=args.clear)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
