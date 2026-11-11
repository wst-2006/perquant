import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.ml_lab.v8_research import V8ResearchConfig, V8ResearchHarness  # noqa: E402
from src.quant_trading.ml_lab.v8_service import V8InferenceService  # noqa: E402
from src.quant_trading.ml_lab.v8_audit import V8DatasetAuditor  # noqa: E402
from src.quant_trading.ml_lab.v8_market_gate import V8MarketGateConfig, V8MarketGateResearch  # noqa: E402
from src.quant_trading.ml_lab.v10_market_gate import V10MarketGateConfig, V10MarketGateResearch  # noqa: E402
from src.quant_trading.ml_lab.v8_events import AnnouncementEventStore, V8EventDatasetBuilder  # noqa: E402
from src.quant_trading.ml_lab.v9_events import V9DailyEventSnapshotBuilder, V9EventDatasetBuilder, V9EventStore  # noqa: E402
from src.quant_trading.ml_lab.v8_daily import V8DailyPipeline  # noqa: E402
from src.quant_trading.ml_lab.v8_labels import V8MarketPanel  # noqa: E402
from src.quant_trading.ml_lab.short_execution import ShortExecutionLabelAugmenter  # noqa: E402
from src.quant_trading.ml_lab.short_execution_audit import ShortExecutionOOSAuditor  # noqa: E402
from src.quant_trading.ml_lab.short_execution_research import (  # noqa: E402
    ShortExecutionResearchConfig,
    ShortExecutionResearchHarness,
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def add_research_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-windows", default="120,240")
    parser.add_argument("--validation-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--models", default="lightgbm,xgboost,catboost")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dataset-audit-path", default="reports/ml_lab/v8_dataset_audit/v8_dataset_audit.json")
    parser.add_argument("--allow-unaudited-dataset", action="store_true")


def config_from_args(args: argparse.Namespace) -> V8ResearchConfig:
    return V8ResearchConfig(
        train_windows=parse_ints(args.train_windows),
        validation_days=args.validation_days,
        step_days=args.step_days,
        models=parse_strings(args.models),
        device=args.device,
        n_estimators=args.n_estimators,
        max_folds=args.max_folds,
        dataset_audit_path=args.dataset_audit_path,
        require_dataset_audit=not args.allow_unaudited_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v8 Top5 opportunity-retrieval research and inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    dataset_audit = sub.add_parser("audit-dataset")
    dataset_audit.add_argument("--dataset-dir", default="reports/ml_lab/training_frame_v8")
    dataset_audit.add_argument("--output-dir", default="reports/ml_lab/v8_dataset_audit")
    dataset_audit.add_argument("--sample-size", type=int, default=10_000)
    dataset_audit.add_argument("--random-state", type=int, default=42)
    dataset_audit.add_argument("--tolerance", type=float, default=5e-6)
    market_gate = sub.add_parser("market-gate-research")
    market_gate.add_argument("--ranking-report", default="reports/ml_lab/v8_1_research_gpu_full/v8_research_report.json")
    market_gate.add_argument("--output-dir", default="reports/ml_lab/v8_market_gate")
    market_gate.add_argument("--target-coverages", default="0.25,0.40,0.55")
    market_gate.add_argument("--min-coverage", type=float, default=0.25)
    v10_market_gate = sub.add_parser("v10-market-gate-research")
    v10_market_gate.add_argument("--ranking-report", default="reports/ml_lab/v9_events_research_gpu_lgb_xgb_full/v8_research_report.json")
    v10_market_gate.add_argument("--output-dir", default="reports/ml_lab/v10_market_gate")
    v10_market_gate.add_argument("--stable-coverages", default="0.25,0.40,0.55")
    v10_market_gate.add_argument("--aggressive-coverages", default="0.00,0.10,0.20")
    v10_market_gate.add_argument("--models", default="rule,logistic,lightgbm")
    v10_market_gate.add_argument("--min-coverage", type=float, default=0.25)
    v10_market_gate.add_argument("--raw-state-factor-set", default="factor_lab_v2")
    augment_events = sub.add_parser("augment-events")
    augment_events.add_argument("--source-dir", default="reports/ml_lab/training_frame_v8")
    augment_events.add_argument("--output-dir", default="reports/ml_lab/training_frame_v8_events")
    augment_v9_events = sub.add_parser("augment-v9-events")
    augment_v9_events.add_argument("--source-dir", default="reports/ml_lab/training_frame_v8")
    augment_v9_events.add_argument("--output-dir", default="reports/ml_lab/training_frame_v9_events")
    augment_v9_events.add_argument("--end-date", default="20260601", help="Maximum public date to load from raw v9 events.")
    snapshot_v9_events = sub.add_parser("materialize-v9-event-snapshots")
    snapshot_v9_events.add_argument("--start-date", required=True)
    snapshot_v9_events.add_argument("--end-date", required=True)
    snapshot_v9_events.add_argument("--factor-set", default="factor_lab_v2")
    snapshot_v9_events.add_argument("--output-dir", required=True)
    augment_short = sub.add_parser("augment-short-labels")
    augment_short.add_argument("--source-dir", default="reports/ml_lab/training_frame_v9_events")
    augment_short.add_argument("--output-dir", default="reports/ml_lab/short_execution_labels_v1")
    augment_short.add_argument("--start-date", default="20240401")
    augment_short.add_argument("--end-date", default=None)
    augment_short.add_argument("--max-exit-scan-days", type=int, default=10)
    audit_short = sub.add_parser("audit-short-execution")
    audit_short.add_argument("--selection-path", default="reports/ml_lab/v11_dual_research_gpu_lgb_xgb_full/v8_oos_top5.csv")
    audit_short.add_argument("--label-dir", default="reports/ml_lab/short_execution_labels_v1")
    audit_short.add_argument("--feature-dir", default="reports/ml_lab/training_frame_v9_events")
    audit_short.add_argument("--output-dir", default="reports/ml_lab/short_execution_oos_audit_v1")
    audit_short.add_argument("--ranking-report", default="reports/ml_lab/v11_dual_research_gpu_lgb_xgb_full/v8_research_report.json")
    audit_short.add_argument("--market-gate-daily", default="reports/ml_lab/v10_market_gate_v11_dual_oos/v10_market_gate_final_daily.csv")
    short_research = sub.add_parser("short-execution-research")
    short_research.add_argument("--feature-dir", default="reports/ml_lab/training_frame_v9_events")
    short_research.add_argument("--label-dir", default="reports/ml_lab/short_execution_labels_v1")
    short_research.add_argument("--output-dir", default="reports/ml_lab/short_execution_research_v1")
    short_research.add_argument("--train-windows", default="120")
    short_research.add_argument("--validation-days", type=int, default=20)
    short_research.add_argument("--step-days", type=int, default=20)
    short_research.add_argument("--models", default="lightgbm")
    short_research.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    short_research.add_argument("--n-estimators", type=int, default=120)
    short_research.add_argument("--max-folds", type=int, default=None)
    research = sub.add_parser("research")
    research.add_argument("--dataset-dir", default="reports/ml_lab/training_frame_v8")
    research.add_argument("--output-dir", default="reports/ml_lab/v8_research")
    add_research_config(research)
    artifact = sub.add_parser("train-artifact")
    artifact.add_argument("--dataset-dir", default="reports/ml_lab/training_frame_v8")
    artifact.add_argument("--research-report", default="reports/ml_lab/v8_research/v8_research_report.json")
    artifact.add_argument("--output-dir", default="reports/ml_lab/v8_artifact")
    artifact.add_argument("--market-gate-report", default=None)
    add_research_config(artifact)
    score = sub.add_parser("score-date")
    score.add_argument("--trade-date", required=True)
    score.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    score.add_argument("--no-persist", action="store_true")
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--signal-date", required=True)
    candidates.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    settle = sub.add_parser("settle")
    settle.add_argument("--as-of-date", required=True)
    settle.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    status = sub.add_parser("status")
    status.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    review = sub.add_parser("manual-review")
    review.add_argument("--signal-id", required=True, type=int)
    review.add_argument("--selected", required=True, choices=["true", "false"])
    review.add_argument("--reason", default=None)
    review.add_argument("--notes", default=None)
    review.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    daily_run = sub.add_parser("daily-run")
    daily_run.add_argument("--trade-date", required=True)
    daily_run.add_argument("--artifact-path", default="reports/ml_lab/v8_artifact/artifact.json")
    daily_run.add_argument("--report-dir", default="reports/ml_lab/v8_daily")
    daily_run.add_argument("--foundation-start-date", default="20240401")
    daily_run.add_argument("--skip-sync", action="store_true", help="Use only already-synchronized PostgreSQL data; all gates still run.")
    args = parser.parse_args()

    harness = V8ResearchHarness()
    if args.command == "audit-dataset":
        result = V8DatasetAuditor().run(
            args.dataset_dir,
            args.output_dir,
            sample_size=args.sample_size,
            random_state=args.random_state,
            tolerance=args.tolerance,
        )
    elif args.command == "market-gate-research":
        result = V8MarketGateResearch().run(
            args.ranking_report,
            args.output_dir,
            V8MarketGateConfig(target_coverages=tuple(float(value) for value in args.target_coverages.split(",")), min_coverage=args.min_coverage),
        )
    elif args.command == "v10-market-gate-research":
        result = V10MarketGateResearch().run(
            args.ranking_report,
            args.output_dir,
            V10MarketGateConfig(
                stable_coverages=tuple(float(value) for value in args.stable_coverages.split(",") if value.strip()),
                aggressive_coverages=tuple(float(value) for value in args.aggressive_coverages.split(",") if value.strip()),
                candidate_models=tuple(value.strip() for value in args.models.split(",") if value.strip()),
                min_coverage=args.min_coverage,
            ),
            raw_state_factor_set=args.raw_state_factor_set,
        )
    elif args.command == "augment-events":
        result = V8EventDatasetBuilder(AnnouncementEventStore.from_database()).build(
            args.source_dir,
            args.output_dir,
        )
    elif args.command == "augment-v9-events":
        result = V9EventDatasetBuilder(V9EventStore.from_database(max_public_date=args.end_date)).build(
            args.source_dir,
            args.output_dir,
        )
    elif args.command == "materialize-v9-event-snapshots":
        store = V9EventStore.from_database_window(start_date=args.start_date, end_date=args.end_date)
        result = V9DailyEventSnapshotBuilder(store, factor_set=args.factor_set).build(
            args.start_date,
            args.end_date,
            args.output_dir,
        )
    elif args.command == "augment-short-labels":
        result = ShortExecutionLabelAugmenter(
            V8MarketPanel.from_database(start_date=args.start_date, end_date=args.end_date),
            max_exit_scan_days=args.max_exit_scan_days,
        ).build(args.source_dir, args.output_dir)
    elif args.command == "audit-short-execution":
        result = ShortExecutionOOSAuditor().run(
            args.selection_path,
            args.label_dir,
            args.feature_dir,
            args.output_dir,
            ranking_report_path=args.ranking_report,
            market_gate_daily_path=args.market_gate_daily,
        )
    elif args.command == "short-execution-research":
        result = ShortExecutionResearchHarness().run(
            args.feature_dir,
            args.label_dir,
            args.output_dir,
            ShortExecutionResearchConfig(
                train_windows=parse_ints(args.train_windows),
                validation_days=args.validation_days,
                step_days=args.step_days,
                models=parse_strings(args.models),
                device=args.device,
                n_estimators=args.n_estimators,
                max_folds=args.max_folds,
            ),
        )
    elif args.command == "research":
        result = harness.run(args.dataset_dir, args.output_dir, config_from_args(args))
    elif args.command == "train-artifact":
        result = harness.train_final_artifact(
            args.dataset_dir,
            args.research_report,
            args.output_dir,
            config_from_args(args),
            market_gate_report_path=args.market_gate_report,
        )
    elif args.command == "daily-run":
        result = V8DailyPipeline(
            artifact_path=args.artifact_path,
            report_dir=args.report_dir,
            foundation_start_date=args.foundation_start_date,
        ).run(args.trade_date, sync=not args.skip_sync)
    else:
        service = V8InferenceService(args.artifact_path)
        if args.command == "score-date":
            result = service.score_date(args.trade_date, persist=not args.no_persist)
        elif args.command == "candidates":
            result = service.get_candidates(args.signal_date)
        elif args.command == "settle":
            result = service.settle(args.as_of_date)
        elif args.command == "status":
            result = service.status()
        else:
            result = service.record_manual_review(
                args.signal_id,
                selected=args.selected == "true",
                reason=args.reason,
                notes=args.notes,
            )
    if args.command == "research":
        result = {
            "task": result["task"],
            "report_contract": result["report_contract"],
            "dataset_audit": result["dataset_audit"],
            "config": result["config"],
            "feature_count": result["feature_count"],
            "model_selection_partition": result["model_selection_partition"],
            "model_selection_summary": result["model_selection_summary"],
            "final_audit": result["final_audit"],
            "final_audit_baseline_summary": result["final_audit_baseline_summary"],
            "gate": result["gate"],
            "report_path": result["report_path"],
            "summary_path": result["summary_path"],
            "oos_daily_path": result["oos_daily_path"],
            "oos_selection_path": result["oos_selection_path"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
