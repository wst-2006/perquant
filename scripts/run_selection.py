import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.selection import SelectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-market A-share stock selection.")
    parser.add_argument("--strategy", default="short_term_core", choices=["short_term_core", "aggressive"])
    parser.add_argument("--trade-date", default=None, help="YYYYMMDD, defaults to today")
    args = parser.parse_args()

    result = SelectionService.create().run_full_market_selection(strategy_type=args.strategy, end_date=args.trade_date)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
