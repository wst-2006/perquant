import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ml_experiment_manifest import build_manifest
from src.quant_trading.ml_lab.service import MlLabConfig, MlLabService
from src.quant_trading.ml_lab.date_splits import split_trade_dates
from src.quant_trading.ml_lab.research import MlResearchHarness, WalkForwardConfig
from src.quant_trading.ml_lab.market_state import MarketStateConfig, MarketStateResearch
from src.quant_trading.ml_lab.paper_observation import PaperObservationService
from src.quant_trading.services.data_sync import DataSyncService
from src.quant_trading.services.foundation_gate import FoundationGateConfig, FoundationGateError, assert_diagnostic_foundation_ready, assert_foundation_ready


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def add_factor_set_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--factor-set", default="factor_lab_v1")


def add_feature_set_arg(parser: argparse.ArgumentParser, default: str = "factor_lab_v2") -> None:
    parser.add_argument("--feature-set", default=default)


def add_snapshot_rebuild_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rebuild-snapshots",
        action="store_true",
        help="Recompute Factor Lab snapshots for the requested feature set before exporting the frame.",
    )


def add_report_contract_arg(parser: argparse.ArgumentParser, default: str = "ml_research_contract_v2") -> None:
    parser.add_argument("--report-contract-version", default=default)


def add_foundation_gate_args(parser: argparse.ArgumentParser, *, gate_mode_default: str = "production") -> None:
    parser.add_argument("--foundation-start-date", default="20240401")
    parser.add_argument("--foundation-end-date", default="20260601")
    parser.add_argument("--gate-mode", choices=["production", "diagnostic"], default=gate_mode_default)
    parser.add_argument(
        "--gate-factor-set",
        default=None,
        help="Feature dependency profile for diagnostic gate. Defaults to --feature-set, then --factor-set, then factor_lab_v2.",
    )
    parser.add_argument(
        "--allow-blocked-foundation",
        action="store_true",
        help="Explicit diagnostic override. Outputs must not be promoted when this is used.",
    )


def resolved_factor_set_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "feature_set", None) or getattr(args, "factor_set", None) or "factor_lab_v2"


def gate_factor_set_from_args(args: argparse.Namespace) -> str:
    return args.gate_factor_set or resolved_factor_set_from_args(args) or "factor_lab_v2"


def foundation_gate_from_args(args: argparse.Namespace, *, allow_diagnostic: bool = True) -> dict:
    config = FoundationGateConfig(start_date=args.foundation_start_date, end_date=args.foundation_end_date)
    if args.gate_mode == "diagnostic":
        if not allow_diagnostic:
            raise FoundationGateError("This command writes promotable artifacts and requires --gate-mode production.")
        return assert_diagnostic_foundation_ready(config, factor_set=gate_factor_set_from_args(args))
    return assert_foundation_ready(
        config,
        allow_blocked=args.allow_blocked_foundation,
    )


def promotion_metadata(gate: dict) -> dict:
    diagnostic_only = bool(gate.get("diagnostic_only") or gate.get("override_used"))
    return {
        "run_mode": "diagnostic" if diagnostic_only else "production",
        "diagnostic_only": diagnostic_only,
        "promotion_allowed": bool(gate.get("promotion_allowed", gate.get("gate_status") == "pass")) and not diagnostic_only,
        "promotion_policy": (
            "Do not promote this output while diagnostic_only=true or while the production foundation gate is blocked."
        ),
    }


def annotate_result(result: dict, gate: dict) -> dict:
    return {**result, **promotion_metadata(gate), "foundation_gate": gate}


def annotate_json_artifact(path: str | Path | None, metadata: dict) -> None:
    if not path:
        return
    artifact_path = Path(path)
    if not artifact_path.exists():
        return
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.update(metadata)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def training_frame_preflight(training_frame_path: str | Path, split_config: WalkForwardConfig | None = None) -> dict:
    universe_audit = DataSyncService(client=None).training_frame_universe_audit(training_frame_path)
    if universe_audit["status"] != "pass":
        raise FoundationGateError(f"Training frame universe audit failed: {universe_audit['issues']}")
    frame = pd.read_csv(training_frame_path, nrows=split_config.max_rows if split_config else None)
    label_audit = MlResearchHarness.audit_training_frame(frame)
    if label_audit["status"] != "pass":
        raise FoundationGateError(f"Training frame label/split audit failed: {label_audit['issues']}")
    feature_cols = sorted([col for col in frame.columns if col.startswith("feature_")])
    feature_missingness_gate = MlLabService.feature_missingness_gate(frame, feature_cols)
    if feature_missingness_gate["status"] != "pass":
        raise FoundationGateError(
            f"Training frame feature missingness gate failed: {feature_missingness_gate['blocking_features']}"
        )
    result = {
        "universe_audit": universe_audit,
        "label_audit": label_audit,
        "feature_missingness_gate": feature_missingness_gate,
    }
    if split_config and split_config.max_rows is not None:
        result["max_rows_preflight_note"] = (
            "Feature missingness was checked on the requested max_rows sample only; run without max_rows for promotion."
        )
    if split_config is not None:
        tradable_frame = MlResearchHarness.load_training_frame(training_frame_path, max_rows=split_config.max_rows)
        folds = MlResearchHarness.walk_forward_windows(tradable_frame["trade_date"], split_config)
        if not folds:
            raise FoundationGateError("Training frame split audit failed: no_walk_forward_folds")
        split_audit = MlResearchHarness.audit_walk_forward_splits(tradable_frame, folds)
        if split_audit["status"] != "pass":
            raise FoundationGateError(f"Training frame split audit failed: {split_audit['issues']}")
        result["split_audit"] = split_audit
    return result


def build_preflight_report(
    training_frame_path: str | Path,
    gate: dict,
    split_config: WalkForwardConfig,
    *,
    report_context: dict | None = None,
) -> dict:
    preflight = training_frame_preflight(training_frame_path, split_config)
    issues = []
    for key in ("universe_audit", "label_audit", "split_audit"):
        audit = preflight.get(key) or {}
        if audit.get("status") != "pass":
            issues.extend(audit.get("issues") or [f"{key}_not_pass"])
    payload = {
        "task": "diagnostic_preflight",
        "training_frame_path": str(training_frame_path),
        "config": {
            "train_days": split_config.train_days,
            "validation_days": split_config.validation_days,
            "step_days": split_config.step_days,
            "top_n": list(split_config.top_n),
            "max_rows": split_config.max_rows,
            "random_state": split_config.random_state,
            "diagnostic_sample_rows": split_config.diagnostic_sample_rows,
            "diagnostic_repeats": split_config.diagnostic_repeats,
            "diagnostic_top_n": split_config.diagnostic_top_n,
        },
        **promotion_metadata(gate),
        "foundation_gate": gate,
        "training_frame_preflight": preflight,
        "status": "pass" if not issues else "review",
        "issues": issues,
    }
    if report_context:
        payload.update(report_context)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ML Lab dual-score research tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    readiness = sub.add_parser("readiness")
    readiness.add_argument("--start-date", default="20240101")
    readiness.add_argument("--end-date", default="20260601")
    add_factor_set_arg(readiness)

    score = sub.add_parser("score-date")
    score.add_argument("--trade-date", required=True)
    add_factor_set_arg(score)

    snapshots = sub.add_parser("build-missing-snapshots")
    snapshots.add_argument("--start-date", required=True)
    snapshots.add_argument("--end-date", required=True)
    snapshots.add_argument("--limit", type=int, default=None)
    snapshots.add_argument("--force", action="store_true")
    add_factor_set_arg(snapshots)

    top10 = sub.add_parser("top10")
    top10.add_argument("--trade-date", required=True)
    add_factor_set_arg(top10)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--start-date", required=True)
    backtest.add_argument("--end-date", required=True)
    backtest.add_argument("--stable-count", type=int, default=5)
    backtest.add_argument("--aggressive-count", type=int, default=2)
    add_factor_set_arg(backtest)

    export = sub.add_parser("export-training-frame")
    export.add_argument("--start-date", required=True)
    export.add_argument("--end-date", required=True)
    export.add_argument("--output-path", default="reports/ml_lab/training_frame.csv")
    export.add_argument("--date-split-preset", default=None)
    export.add_argument("--split-manifest-path", default=None)
    export.add_argument("--data-manifest-path", default=None)
    add_factor_set_arg(export)
    add_feature_set_arg(export)
    add_snapshot_rebuild_arg(export)
    add_report_contract_arg(export)
    add_foundation_gate_args(export)

    train = sub.add_parser("train-models")
    train.add_argument("--training-frame-path", required=True)
    train.add_argument("--output-dir", default="reports/ml_lab/models")
    train.add_argument("--factor-pool-version", default=None)
    add_feature_set_arg(train)
    add_report_contract_arg(train)
    add_foundation_gate_args(train)

    convergence = sub.add_parser("factor-convergence")
    convergence.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    convergence.add_argument("--output-dir", default="reports/ml_lab/convergence")
    convergence.add_argument("--model-report-path", default="reports/ml_lab/models_full/model_comparison_report.json")
    convergence.add_argument("--factor-pool-version", default=None)

    research = sub.add_parser("research-harness")
    research.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    research.add_argument("--output-dir", default="reports/ml_lab/research_harness")
    research.add_argument("--train-days", type=int, default=120)
    research.add_argument("--validation-days", type=int, default=20)
    research.add_argument("--step-days", type=int, default=20)
    research.add_argument("--top-n", default="10,20,50")
    research.add_argument("--models", default="lightgbm")
    research.add_argument("--max-rows", type=int, default=None)
    add_feature_set_arg(research)
    add_report_contract_arg(research)
    add_foundation_gate_args(research)

    preflight = sub.add_parser("diagnostic-preflight")
    preflight.add_argument("--training-frame-path", required=True)
    preflight.add_argument("--output-path", default="reports/ml_lab/diagnostic_preflight.json")
    preflight.add_argument("--train-days", type=int, default=120)
    preflight.add_argument("--validation-days", type=int, default=20)
    preflight.add_argument("--step-days", type=int, default=20)
    preflight.add_argument("--top-n", default="10,20,50")
    preflight.add_argument("--max-rows", type=int, default=None)
    add_feature_set_arg(preflight)
    add_report_contract_arg(preflight)
    add_foundation_gate_args(preflight, gate_mode_default="diagnostic")

    audit = sub.add_parser("audit-research")
    audit.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    audit.add_argument("--output-dir", default="reports/ml_lab/research_audit")
    audit.add_argument("--model-report-path", default="reports/ml_lab/research_harness_v1_full/research_harness_report.json")
    audit.add_argument("--train-days", type=int, default=120)
    audit.add_argument("--validation-days", type=int, default=20)
    audit.add_argument("--step-days", type=int, default=20)
    audit.add_argument("--top-n", default="10,20,50")
    audit.add_argument("--max-rows", type=int, default=None)
    add_feature_set_arg(audit)
    add_report_contract_arg(audit)
    add_foundation_gate_args(audit, gate_mode_default="diagnostic")

    market_state = sub.add_parser("market-state-research")
    market_state.add_argument("--ranking-report-path", required=True)
    market_state.add_argument("--baseline-audit-path", required=True)
    market_state.add_argument("--output-dir", default="reports/ml_lab/market_state")
    market_state.add_argument("--target-coverage", type=float, default=0.25)
    market_state.add_argument("--min-coverage", type=float, default=0.25)
    market_state.add_argument("--min-history-days", type=int, default=60)
    market_state.add_argument("--raw-state-factor-set", default="factor_lab_v2")
    add_feature_set_arg(market_state, default="factor_lab_v5")
    add_report_contract_arg(market_state)
    add_foundation_gate_args(market_state, gate_mode_default="diagnostic")

    paper_observation = sub.add_parser("paper-observation")
    paper_observation.add_argument("--signal-date", required=True)
    paper_observation.add_argument("--observation-date", required=True)
    paper_observation.add_argument("--output-dir", default="reports/ml_lab/paper_observation")
    paper_observation.add_argument("--foundation-start-date", default="20240401")
    add_factor_set_arg(paper_observation)

    label_audit = sub.add_parser("audit-label-reconstruction")
    label_audit.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    label_audit.add_argument("--output-dir", default="reports/ml_lab/label_reconstruction_audit")
    label_audit.add_argument("--sample-size", type=int, default=1000)
    label_audit.add_argument("--random-state", type=int, default=42)
    label_audit.add_argument("--tolerance", type=float, default=1e-9)

    publish = sub.add_parser("publish-versioned-artifacts")
    publish.add_argument("--training-frame-path", default="reports/ml_lab/training_frame_full.csv")
    publish.add_argument("--models-output-dir", default="reports/ml_lab/models_versioned_v2")
    publish.add_argument("--convergence-output-dir", default="reports/ml_lab/convergence_versioned_v2")
    publish.add_argument("--model-report-path", default="reports/ml_lab/models_versioned_v2/model_comparison_report.json")
    publish.add_argument("--factor-pool-version", default=None)
    publish.add_argument("--package-output-path", default="reports/ml_lab/execution_package_v1.json")
    publish.add_argument("--registry-output-path", default="reports/ml_lab/versioned_registry.json")
    add_feature_set_arg(publish)
    add_report_contract_arg(publish)
    add_foundation_gate_args(publish)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = MlLabService(MlLabConfig(factor_set=resolved_factor_set_from_args(args), feature_set=resolved_factor_set_from_args(args)))
    if args.command == "readiness":
        result = service.readiness(start_date=args.start_date, end_date=args.end_date)
    elif args.command == "build-missing-snapshots":
        result = service.build_missing_snapshots(args.start_date, args.end_date, limit=args.limit, force=args.force)
    elif args.command == "score-date":
        result = service.score_date(args.trade_date)
    elif args.command == "top10":
        result = service.get_top10(args.trade_date)
    elif args.command == "backtest":
        result = service.backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            stable_count=args.stable_count,
            aggressive_count=args.aggressive_count,
        )
    elif args.command == "export-training-frame":
        gate = foundation_gate_from_args(args)
        generation_factor_set = resolved_factor_set_from_args(args)
        trade_dates = None
        split_manifest = None
        if args.date_split_preset:
            all_dates = service._trade_dates(args.start_date, args.end_date)
            split_manifest = split_trade_dates(all_dates, args.date_split_preset)
            trade_dates = split_manifest["train_dates"]
        if args.rebuild_snapshots:
            snapshot_result = service.build_missing_snapshots(
                args.start_date,
                args.end_date,
                force=True,
            )
            if snapshot_result.get("failed"):
                raise FoundationGateError(f"Snapshot rebuild failed: {snapshot_result}")
        result = service.export_training_frame(args.start_date, args.end_date, args.output_path, trade_dates=trade_dates)
        universe_audit = DataSyncService(client=None).training_frame_universe_audit(args.output_path)
        if universe_audit["status"] != "pass":
            raise FoundationGateError(f"Exported training frame universe audit failed: {universe_audit['issues']}")
        manifest_path = Path(args.data_manifest_path or Path(args.output_path).with_suffix(".data_dependency_manifest.json"))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data_manifest = build_manifest(
            args.output_path,
            {
                "factor_set": generation_factor_set,
                "feature_set": generation_factor_set,
                "requested_factor_set": args.factor_set,
                "requested_feature_set": args.feature_set,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "date_split_preset": args.date_split_preset,
                "report_contract_version": args.report_contract_version,
            },
            notes="generated by export-training-frame; no secrets included",
            run_mode=promotion_metadata(gate)["run_mode"],
            foundation_gate=gate,
            data_dependencies=gate.get("data_dependency_profile"),
            training_frame_universe_audit=universe_audit,
            feature_contract=result.get("feature_contract"),
        )
        manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        result = annotate_result(result, gate)
        result["factor_set"] = generation_factor_set
        result["feature_set"] = generation_factor_set
        result["requested_factor_set"] = args.factor_set
        result["requested_feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
        result["data_dependency_manifest_path"] = str(manifest_path)
        result["training_frame_universe_audit"] = universe_audit
        if args.rebuild_snapshots:
            result["snapshot_rebuild"] = snapshot_result
        if split_manifest:
            result["date_split"] = {
                key: value
                for key, value in split_manifest.items()
                if key not in {"train_dates", "holdout_dates"}
            }
            result["train_trade_days"] = split_manifest["train_trade_days"]
            result["holdout_trade_days"] = split_manifest["holdout_trade_days"]
            manifest_path = Path(args.split_manifest_path or Path(args.output_path).with_suffix(".split_manifest.json"))
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            result["split_manifest_path"] = str(manifest_path)
    elif args.command == "train-models":
        gate = foundation_gate_from_args(args)
        preflight = training_frame_preflight(args.training_frame_path)
        result = service.train_models(args.training_frame_path, args.output_dir, factor_pool_version=args.factor_pool_version)
        result = annotate_result(result, gate)
        result["feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
        result["training_frame_preflight"] = preflight
        annotate_json_artifact(
            result.get("report_path"),
            {
                **promotion_metadata(gate),
                "foundation_gate": gate,
                "training_frame_preflight": preflight,
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
    elif args.command == "factor-convergence":
        result = service.build_factor_convergence_report(args.training_frame_path, args.output_dir, args.model_report_path, factor_pool_version=args.factor_pool_version)
    elif args.command == "research-harness":
        gate = foundation_gate_from_args(args)
        config = WalkForwardConfig(
            train_days=args.train_days,
            validation_days=args.validation_days,
            step_days=args.step_days,
            top_n=parse_int_tuple(args.top_n),
            base_models=parse_str_tuple(args.models),
            max_rows=args.max_rows,
        )
        preflight = training_frame_preflight(args.training_frame_path, config)
        result = MlResearchHarness().run_from_training_frame(
            training_frame_path=args.training_frame_path,
            output_dir=args.output_dir,
            config=config,
            report_context={
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
        result = annotate_result(result, gate)
        result["feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
        result["training_frame_preflight"] = preflight
        annotate_json_artifact(
            result.get("report_path"),
            {
                **promotion_metadata(gate),
                "foundation_gate": gate,
                "training_frame_preflight": preflight,
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
    elif args.command == "diagnostic-preflight":
        gate = foundation_gate_from_args(args)
        config = WalkForwardConfig(
            train_days=args.train_days,
            validation_days=args.validation_days,
            step_days=args.step_days,
            top_n=parse_int_tuple(args.top_n),
            base_models=(),
            max_rows=args.max_rows,
        )
        result = build_preflight_report(
            args.training_frame_path,
            gate,
            config,
            report_context={
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        result["report_path"] = str(output_path)
    elif args.command == "audit-research":
        gate = foundation_gate_from_args(args)
        config = WalkForwardConfig(
            train_days=args.train_days,
            validation_days=args.validation_days,
            step_days=args.step_days,
            top_n=parse_int_tuple(args.top_n),
            base_models=(),
            max_rows=args.max_rows,
        )
        result = MlResearchHarness().run_audit_from_training_frame(
            training_frame_path=args.training_frame_path,
            output_dir=args.output_dir,
            config=config,
            model_report_path=args.model_report_path,
            report_context={
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
        result = annotate_result(result, gate)
        result["status"] = result.get("gate", {}).get("status")
        result["feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
        annotate_json_artifact(
            result.get("report_path"),
            {
                **promotion_metadata(gate),
                "foundation_gate": gate,
                "status": result.get("gate", {}).get("status"),
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
    elif args.command == "market-state-research":
        gate = foundation_gate_from_args(args)
        config = MarketStateConfig(
            target_coverage=args.target_coverage,
            min_coverage=args.min_coverage,
            min_history_days=args.min_history_days,
        )
        result = MarketStateResearch().run(
            ranking_report_path=args.ranking_report_path,
            baseline_audit_path=args.baseline_audit_path,
            output_dir=args.output_dir,
            config=config,
            raw_state_factor_set=args.raw_state_factor_set,
            report_context={
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
            },
        )
        result = annotate_result(result, gate)
        result["feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
        annotate_json_artifact(
            result.get("report_path"),
            {
                **promotion_metadata(gate),
                "foundation_gate": gate,
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
                "gate_status": result.get("gate", {}).get("status"),
            },
        )
    elif args.command == "paper-observation":
        result = PaperObservationService().run(
            signal_date=args.signal_date,
            observation_date=args.observation_date,
            factor_set=args.factor_set,
            model_version=service.config.model_version,
            output_dir=args.output_dir,
            foundation_start_date=args.foundation_start_date,
        )
    elif args.command == "audit-label-reconstruction":
        result = MlResearchHarness().run_label_reconstruction_audit(
            training_frame_path=args.training_frame_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            random_state=args.random_state,
            tolerance=args.tolerance,
        )
    elif args.command == "publish-versioned-artifacts":
        gate = foundation_gate_from_args(args, allow_diagnostic=False)
        preflight = training_frame_preflight(args.training_frame_path)
        training_report = service.train_models(args.training_frame_path, args.models_output_dir, factor_pool_version=args.factor_pool_version)
        training_report_path = Path(args.models_output_dir) / "model_comparison_report.json"
        annotate_json_artifact(
            training_report_path,
            {
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
                "foundation_gate": gate,
                "training_frame_preflight": preflight,
                **promotion_metadata(gate),
            },
        )
        convergence_report = service.build_factor_convergence_report(
            args.training_frame_path,
            args.convergence_output_dir,
            args.model_report_path,
            factor_pool_version=args.factor_pool_version,
        )
        convergence_report_path = Path(args.convergence_output_dir) / "factor_convergence_details.json"
        annotate_json_artifact(
            convergence_report_path,
            {
                "feature_set": args.feature_set,
                "report_contract_version": args.report_contract_version,
                "foundation_gate": gate,
                "training_frame_preflight": preflight,
                **promotion_metadata(gate),
            },
        )
        package_path = Path(args.package_output_path)
        registry_path = Path(args.registry_output_path)
        package = service.build_execution_package(
            factor_pool_version=args.factor_pool_version or training_report.get("factor_pool_version") or service.config.factor_set,
            training_report_path=training_report_path,
            convergence_report_path=convergence_report_path,
            registry_path=registry_path,
            output_path=package_path,
        )
        registry = service.write_versioned_registry(
            output_path=registry_path,
            factor_pool_version=package["factor_pool_version"],
            factor_pool_run_id=None,
            ml_training_report=training_report_path,
            ml_validation_summary=Path(args.models_output_dir) / "model_validation_summary.json",
            ml_experiment_ledger=Path(args.models_output_dir) / "experiment_ledger.json",
            factor_convergence_report=convergence_report_path,
            factor_convergence_ledger=Path(args.convergence_output_dir) / "experiment_ledger.json",
        )
        result = annotate_result(
            {
                "training_report": training_report,
                "convergence_report": convergence_report,
                "package": package,
                "registry": registry,
                "training_frame_preflight": preflight,
            },
            gate,
        )
        result["feature_set"] = args.feature_set
        result["report_contract_version"] = args.report_contract_version
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except FoundationGateError as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": "foundation_gate_blocked",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2)
