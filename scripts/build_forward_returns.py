import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.factor_lab.service import FactorLabConfig, FactorLabService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward returns for factor snapshots.")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--horizons", default="1,3,5,10,20", help="Comma-separated trading-day horizons")
    parser.add_argument("--factor-set", default="factor_lab_v1")
    args = parser.parse_args()

    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    result = FactorLabService(FactorLabConfig(factor_set=args.factor_set)).build_forward_returns(args.trade_date, horizons=horizons)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
