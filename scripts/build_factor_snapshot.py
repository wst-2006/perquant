import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.factor_lab.service import FactorLabConfig, FactorLabService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-market factor snapshots from real local Tushare data.")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--factor-set", default="factor_lab_v1")
    args = parser.parse_args()

    result = FactorLabService(FactorLabConfig(factor_set=args.factor_set)).build_snapshot(args.trade_date)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
