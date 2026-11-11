import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.data_sync import DataSyncService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize real Tushare data into PostgreSQL.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stock-pool")
    sub.add_parser("stock-universe")
    daily = sub.add_parser("daily")
    daily.add_argument("--trade-date", default=None)
    basic = sub.add_parser("daily-basic")
    basic.add_argument("--trade-date", default=None)
    history = sub.add_parser("daily-history")
    history.add_argument("--start-date", required=True)
    history.add_argument("--end-date", required=True)
    basic_history = sub.add_parser("daily-basic-history")
    basic_history.add_argument("--start-date", required=True)
    basic_history.add_argument("--end-date", required=True)
    moneyflow = sub.add_parser("moneyflow-full-market")
    moneyflow.add_argument("--start-date", required=True)
    moneyflow.add_argument("--end-date", required=True)
    adj_factor = sub.add_parser("adj-factor-full-market")
    adj_factor.add_argument("--start-date", required=True)
    adj_factor.add_argument("--end-date", required=True)
    stk_limit = sub.add_parser("stk-limit-batch")
    stk_limit.add_argument("--start-date", required=True)
    stk_limit.add_argument("--end-date", required=True)
    index_daily = sub.add_parser("index-daily")
    index_daily.add_argument("--start-date", required=True)
    index_daily.add_argument("--end-date", required=True)
    index_daily.add_argument("--index-codes", default=None, help="Comma-separated index codes. Defaults to core A-share indexes.")
    index_basic = sub.add_parser("index-basic")
    index_basic.add_argument("--markets", default="all", help="Comma-separated markets or all.")
    index_universe = sub.add_parser("index-daily-universe")
    index_universe.add_argument("--start-date", required=True)
    index_universe.add_argument("--end-date", required=True)
    index_universe.add_argument("--markets", default="all", help="Comma-separated markets or all.")
    index_universe.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit for index codes.")
    index_universe.add_argument("--offset", type=int, default=0)
    index_universe.add_argument("--no-skip-complete", action="store_true")
    financial = sub.add_parser("financial-foundation")
    financial.add_argument("--start-date", required=True)
    financial.add_argument("--end-date", required=True)
    financial.add_argument("--endpoints", default="all", help="Comma-separated endpoints or all.")
    financial.add_argument("--mode", default="auto", choices=["auto", "vip", "stock"])
    financial.add_argument("--offset", type=int, default=0)
    financial.add_argument("--limit", type=int, default=None)
    financial.add_argument("--no-skip-existing", action="store_true")
    v9_events = sub.add_parser("v9-event-foundation")
    v9_events.add_argument("--start-date", required=True)
    v9_events.add_argument("--end-date", required=True)
    v9_events.add_argument("--endpoints", default="all", help="Comma-separated endpoints or all.")
    v9_events.add_argument("--page-size", type=int, default=6000)
    v9_events.add_argument("--max-pages", type=int, default=100)
    v9_status = sub.add_parser("v9-event-status")
    v9_status.add_argument("--start-date", required=True)
    v9_status.add_argument("--end-date", required=True)
    legacy_financial = sub.add_parser("migrate-legacy-raw-financial")
    legacy_financial.add_argument("--endpoints", default="all", help="Comma-separated endpoints or all.")
    legacy_financial.add_argument("--start-date", default=None)
    legacy_financial.add_argument("--end-date", default=None)
    foundation_status = sub.add_parser("foundation-status")
    foundation_status.add_argument("--start-date", required=True)
    foundation_status.add_argument("--end-date", required=True)
    stock_audit = sub.add_parser("stock-universe-audit")
    stock_audit.add_argument("--preview-limit", type=int, default=20)
    quarantine = sub.add_parser("quarantine-non-stock-placeholders")
    quarantine.add_argument("--preview-limit", type=int, default=20)
    quarantine.add_argument("--apply", action="store_true", help="Apply quarantine. Default is dry run.")
    repair_unknown = sub.add_parser("repair-unknown-a-share-metadata")
    repair_unknown.add_argument("--limit", type=int, default=None)
    repair_unknown.add_argument("--apply", action="store_true", help="Apply metadata repair. Default is dry run.")
    batch = sub.add_parser("calendar-batch")
    batch.add_argument("--start-date", required=True)
    batch.add_argument("--end-date", required=True)
    batch.add_argument("--datasets", default="daily,daily_basic,moneyflow", help="Comma-separated: daily,daily_basic,moneyflow,adj_factor,stk_limit,index_daily")
    sub.add_parser("status")

    args = parser.parse_args()
    if args.command == "status":
        result = DataSyncService(client=None).data_status()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "foundation-status":
        result = DataSyncService(client=None).foundation_status(args.start_date, args.end_date)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "stock-universe-audit":
        result = DataSyncService(client=None).stock_universe_audit(preview_limit=args.preview_limit)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "v9-event-status":
        result = DataSyncService(client=None).v9_event_status(args.start_date, args.end_date)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "quarantine-non-stock-placeholders":
        result = DataSyncService(client=None).quarantine_non_stock_placeholders(dry_run=not args.apply, preview_limit=args.preview_limit)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    service = DataSyncService.create()
    if args.command == "stock-pool":
        result = service.sync_stock_pool()
    elif args.command == "stock-universe":
        result = service.sync_stock_universe()
    elif args.command == "daily":
        result = service.sync_daily_quotes(args.trade_date)
    elif args.command == "daily-basic":
        result = service.sync_daily_basic(args.trade_date)
    elif args.command == "daily-history":
        result = service.sync_daily_history_full_market(args.start_date, args.end_date)
    elif args.command == "daily-basic-history":
        result = service.sync_daily_basic_history_full_market(args.start_date, args.end_date)
    elif args.command == "moneyflow-full-market":
        result = service.sync_moneyflow_full_market(args.start_date, args.end_date)
    elif args.command == "adj-factor-full-market":
        result = service.sync_adj_factor_full_market(args.start_date, args.end_date)
    elif args.command == "stk-limit-batch":
        result = service.sync_stk_limit_batch(args.start_date, args.end_date)
    elif args.command == "index-daily":
        index_codes = [item.strip() for item in args.index_codes.split(",") if item.strip()] if args.index_codes else None
        result = service.sync_index_daily(args.start_date, args.end_date, index_codes=index_codes)
    elif args.command == "index-basic":
        markets = None if args.markets == "all" else [item.strip() for item in args.markets.split(",") if item.strip()]
        result = service.sync_index_basic(markets=markets)
    elif args.command == "index-daily-universe":
        markets = None if args.markets == "all" else [item.strip() for item in args.markets.split(",") if item.strip()]
        result = service.sync_index_daily_universe(
            args.start_date,
            args.end_date,
            markets=markets,
            limit=args.limit,
            offset=args.offset,
            skip_complete=not args.no_skip_complete,
        )
    elif args.command == "financial-foundation":
        endpoints = None if args.endpoints == "all" else [item.strip() for item in args.endpoints.split(",") if item.strip()]
        result = service.sync_financial_foundation(
            args.start_date,
            args.end_date,
            endpoints=endpoints,
            mode=args.mode,
            offset=args.offset,
            limit=args.limit,
            skip_existing=not args.no_skip_existing,
        )
    elif args.command == "migrate-legacy-raw-financial":
        endpoints = None if args.endpoints == "all" else [item.strip() for item in args.endpoints.split(",") if item.strip()]
        result = service.migrate_legacy_raw_financial_records(endpoints=endpoints, start_date=args.start_date, end_date=args.end_date)
    elif args.command == "v9-event-foundation":
        endpoints = None if args.endpoints == "all" else [item.strip() for item in args.endpoints.split(",") if item.strip()]
        result = service.sync_v9_event_foundation(
            args.start_date,
            args.end_date,
            endpoints=endpoints,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
    elif args.command == "repair-unknown-a-share-metadata":
        result = service.repair_unknown_a_share_metadata(dry_run=not args.apply, limit=args.limit)
    elif args.command == "calendar-batch":
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
        result = service.sync_calendar_batch(args.start_date, args.end_date, datasets=datasets)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
