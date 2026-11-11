import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.review import ReviewService


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage manual review journal entries.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--title", required=True)
    add.add_argument("--content", required=True)
    add.add_argument("--ts-code", default=None)
    add.add_argument("--name", default=None)
    add.add_argument("--category", default="daily")
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=50)
    list_cmd.add_argument("--ts-code", default=None)

    args = parser.parse_args()
    service = ReviewService()
    if args.command == "add":
        result = {
            "id": service.add_entry(
                title=args.title,
                content=args.content,
                ts_code=args.ts_code,
                name=args.name,
                category=args.category,
            )
        }
    else:
        result = service.list_entries(limit=args.limit, ts_code=args.ts_code)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
