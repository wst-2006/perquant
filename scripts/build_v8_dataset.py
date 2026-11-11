import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.ml_lab.v8_labels import V8DatasetBuilder, V8MarketPanel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real-data, T+1-safe v8 partitioned training dataset.")
    parser.add_argument("--source-path", default="reports/ml_lab/training_frame_v5_full.csv")
    parser.add_argument("--output-dir", default="reports/ml_lab/training_frame_v8")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    source = Path(args.source_path)
    header = pd.read_csv(source, usecols=["trade_date"], nrows=1)
    start_date = str(header.iloc[0, 0]).replace(".0", "")[:8]
    panel = V8MarketPanel.from_database(start_date=start_date)
    result = V8DatasetBuilder(panel).build(source, args.output_dir, chunksize=args.chunksize)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
