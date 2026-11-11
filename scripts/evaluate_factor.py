import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.factor_lab.service import FactorLabService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one factor with Rank IC, quantile returns, and top-N returns.")
    parser.add_argument("--factor", required=True, help="Factor name, for example return_20d")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=5)
    args = parser.parse_args()

    result = FactorLabService().evaluate_factor(
        factor_name=args.factor,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        quantiles=args.quantiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
