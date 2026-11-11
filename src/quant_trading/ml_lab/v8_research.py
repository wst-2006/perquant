"""GPU-capable walk-forward ranker research for the v8 Top5 contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .v8_labels import V8_LABEL_CONTRACT
from .v8_audit import V8DatasetAuditError, V8DatasetAuditor
from .v8_features import V8_LIVE_FEATURE_CONTRACT
from .v8_metrics import evaluate_top5, summarize_folds, v8_promotion_gate
from .market_state import MarketStateResearch
from .research import MARKET_STATE_FEATURES


V8_REPORT_CONTRACT = "ml_v8_top5_research_v3"
V11_DUAL_VARIANT = "v11_dual_70_30"
V8_HEADS = {
    "return_5d": {"kind": "rank", "target": "label_5d_net_return", "weight": 0.30},
    "hit5_10d": {"kind": "classification", "target": "label_10d_hit_5pct", "weight": 0.30},
    "upside_10d": {"kind": "rank", "target": "label_10d_mfe_close", "weight": 0.25},
    "safety_10d": {"kind": "regression", "target": "label_10d_mae_close", "weight": 0.15},
}
V8_SCORE_VARIANTS = ("composite", *V8_HEADS.keys(), V11_DUAL_VARIANT)


class V8ResearchError(RuntimeError):
    """Raised when a v8 research contract cannot be satisfied."""


@dataclass(frozen=True)
class V8ResearchConfig:
    train_windows: tuple[int, ...] = (120, 240)
    validation_days: int = 20
    step_days: int = 20
    max_label_horizon: int = 20
    models: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")
    device: str = "gpu"
    n_estimators: int = 120
    random_state: int = 42
    max_folds: int | None = None
    dataset_audit_path: str | None = None
    require_dataset_audit: bool = False
    final_audit_fraction: float = 0.30
    min_final_audit_folds: int = 3
    permutation_feature_count: int = 5


class V8ParquetDataset:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        if not self.manifest_path.exists():
            raise V8ResearchError(f"Missing v8 manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("label_contract") != V8_LABEL_CONTRACT:
            raise V8ResearchError(f"Unsupported v8 label contract: {self.manifest.get('label_contract')}")
        self.partitions = {
            path.name.split("=", 1)[1]: path / "part-000.parquet"
            for path in self.directory.glob("trade_date=*")
            if (path / "part-000.parquet").exists()
        }
        self.dates = sorted(self.partitions)
        if not self.dates:
            raise V8ResearchError("v8 dataset has no date partitions.")
        sample = pd.read_parquet(self.partitions[self.dates[0]])
        self.columns = sample.columns.tolist()
        market_state_columns = set(MARKET_STATE_FEATURES)
        self.feature_columns = sorted(
            col for col in self.columns if col.startswith("feature_") and col not in market_state_columns
        )
        if not self.feature_columns:
            raise V8ResearchError("v8 dataset has no feature columns.")

    def load_dates(self, dates: list[str], columns: list[str] | None = None) -> pd.DataFrame:
        paths = [self.partitions[date] for date in dates if date in self.partitions]
        if not paths:
            return pd.DataFrame(columns=columns or self.columns)
        return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


class V8ResearchHarness:
    def run(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        config: V8ResearchConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or V8ResearchConfig()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        progress_path = output / "progress.jsonl"
        partial_path = output / "partial_results.json"
        dataset = V8ParquetDataset(dataset_dir)
        dataset_audit = self.load_dataset_audit(dataset_dir, cfg)
        required = self.required_columns(dataset.feature_columns)
        missing = sorted(set(required) - set(dataset.columns))
        if missing:
            raise V8ResearchError(f"v8 dataset missing research columns: {missing}")
        results: list[dict[str, Any]] = []
        baseline_results: dict[str, list[dict[str, Any]]] = {}
        planned_partition = self.planned_validation_partition(dataset.dates, cfg)
        permutation_validation_starts = set(planned_partition.get("audit_validation_starts") or [])
        planned_fold_count = 0
        for train_window in cfg.train_windows:
            folds = self.walk_forward_windows(dataset.dates, train_window, cfg)
            planned_fold_count += len(folds[: cfg.max_folds] if cfg.max_folds is not None else folds)
        self._append_progress(
            progress_path,
            {
                "event": "research_started",
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output),
                "date_count": len(dataset.dates),
                "feature_count": len(dataset.feature_columns),
                "planned_fold_count": planned_fold_count,
                "models": list(cfg.models),
                "config": asdict(cfg),
            },
        )
        self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "running"))
        for train_window in cfg.train_windows:
            folds = self.walk_forward_windows(dataset.dates, train_window, cfg)
            if cfg.max_folds is not None:
                folds = folds[: cfg.max_folds]
            for fold_index, fold in enumerate(folds):
                train_dates, validation_dates = fold["train_dates"], fold["validation_dates"]
                self._append_progress(
                    progress_path,
                    {
                        "event": "fold_started",
                        "train_window": train_window,
                        "fold_index": fold_index,
                        "train_start": train_dates[0],
                        "train_end": train_dates[-1],
                        "validation_start": validation_dates[0],
                        "validation_end": validation_dates[-1],
                    },
                )
                frame = dataset.load_dates(train_dates + validation_dates, required)
                train = frame[frame["trade_date"].astype(str).isin(train_dates)].copy()
                valid = frame[frame["trade_date"].astype(str).isin(validation_dates)].copy()
                train, split_audit = self.apply_embargo(train, validation_dates[0])
                train = self.clean_eligible_rows(train)
                valid = self.clean_eligible_rows(valid)
                if train.empty or valid.empty:
                    results.append({"status": "empty_fold", "train_window": train_window, "fold_index": fold_index, "split_audit": split_audit})
                    self._append_progress(
                        progress_path,
                        {
                            "event": "fold_empty",
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "validation_start": validation_dates[0],
                            "train_rows": int(len(train)),
                            "validation_rows": int(len(valid)),
                        },
                    )
                    self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "running"))
                    continue
                for name, scored in self.baseline_predictions(valid, cfg.random_state).items():
                    baseline_results.setdefault(name, []).append(
                        {
                            "status": "trained",
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "validation_start": validation_dates[0],
                            "validation_end": validation_dates[-1],
                            "metrics": _compact_metrics(evaluate_top5(scored)),
                        }
                    )
                self._append_progress(
                    progress_path,
                    {
                        "event": "baselines_completed",
                        "train_window": train_window,
                        "fold_index": fold_index,
                        "validation_start": validation_dates[0],
                        "baseline_count": len(baseline_results),
                    },
                )
                self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "running"))
                fold_predictions: dict[str, dict[str, np.ndarray]] = {}
                fold_family_results: list[dict[str, Any]] = []
                for model_name in cfg.models:
                    result = self.evaluate_model_family(
                        train,
                        valid,
                        dataset.feature_columns,
                        model_name,
                        cfg,
                        collect_permutation=validation_dates[0] in permutation_validation_starts,
                    )
                    fold_predictions[model_name] = result.pop("_variant_predictions")
                    result.update(
                        {
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "train_start": train_dates[0],
                            "train_end": train_dates[-1],
                            "validation_start": validation_dates[0],
                            "validation_end": validation_dates[-1],
                            "split_audit": split_audit,
                        }
                    )
                    results.append(result)
                    fold_family_results.append(result)
                    self._append_progress(
                        progress_path,
                        {
                            "event": "model_completed",
                            "model": model_name,
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "validation_start": validation_dates[0],
                            "train_rows": result.get("train_rows"),
                            "validation_rows": result.get("validation_rows"),
                            "metrics": result.get("metrics"),
                        },
                    )
                    self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "running"))
                if len(fold_predictions) >= 2:
                    ensemble = self.evaluate_model_ensemble(valid, fold_predictions, fold_family_results)
                    ensemble.update(
                        {
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "train_start": train_dates[0],
                            "train_end": train_dates[-1],
                            "validation_start": validation_dates[0],
                            "validation_end": validation_dates[-1],
                            "split_audit": split_audit,
                        }
                    )
                    results.append(ensemble)
                    self._append_progress(
                        progress_path,
                        {
                            "event": "ensemble_completed",
                            "train_window": train_window,
                            "fold_index": fold_index,
                            "validation_start": validation_dates[0],
                            "metrics": ensemble.get("metrics"),
                        },
                    )
                    self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "running"))

        variant_results = self.flatten_variant_results(results)
        summaries = self.summarize_models(variant_results)
        baseline_summaries = [
            {"baseline": name, **summarize_folds(rows)} for name, rows in sorted(baseline_results.items())
        ]
        evaluation_partition = self.partition_model_selection(variant_results, cfg)
        selection_results = [
            row for row in variant_results if row.get("validation_start") in evaluation_partition["selection_validation_starts"]
        ]
        selection_summaries = self.summarize_models(selection_results)
        best = self.best_summary(selection_summaries or summaries)
        audit_results = [
            row
            for row in variant_results
            if row.get("validation_start") in evaluation_partition["audit_validation_starts"]
            and row.get("model") == (best or {}).get("model")
            and row.get("train_window") == (best or {}).get("train_window")
            and row.get("score_variant") == (best or {}).get("score_variant")
        ]
        final_audit = {
            "model": (best or {}).get("model"),
            "train_window": (best or {}).get("train_window"),
            "score_variant": (best or {}).get("score_variant"),
            **summarize_folds(audit_results),
        }
        audit_baseline_summaries = [
            {
                "baseline": name,
                **summarize_folds(
                    [
                        row
                        for row in rows
                        if row.get("train_window") == (best or {}).get("train_window")
                        and row.get("validation_start") in evaluation_partition["audit_validation_starts"]
                    ]
                ),
            }
            for name, rows in sorted(baseline_results.items())
        ]
        oos_daily, oos_selections = self.best_oos_stream(
            variant_results,
            best,
            allowed_validation_starts=set(evaluation_partition["audit_validation_starts"]),
        )
        regime_summary, regime_guard = self.regime_audit(oos_daily)
        split_audits_passed = bool(audit_results) and all(
            (row.get("split_audit") or {}).get("status") == "pass" for row in audit_results
        )
        audit_chain_passed = (
            dataset_audit.get("status") == "pass"
            and evaluation_partition.get("status") == "pass"
            and split_audits_passed
        )
        gate = v8_promotion_gate(
            final_audit,
            audit_baseline_summaries,
            audits_passed=audit_chain_passed,
            regime_guard_passed=regime_guard,
        )
        failure_windows = self.failure_windows(oos_daily, audit_results)
        feature_diagnostics = self.feature_diagnostics(audit_results)
        payload = {
            "task": "v8_top5_ranker_research",
            "report_contract": V8_REPORT_CONTRACT,
            "dataset_dir": str(dataset_dir),
            "dataset_manifest": dataset.manifest,
            "dataset_audit": dataset_audit,
            "config": asdict(cfg),
            "feature_count": len(dataset.feature_columns),
            "heads": V8_HEADS,
            "score_variants": list(V8_SCORE_VARIANTS),
            "results": self.compact_results_for_report(results),
            "summary": summaries,
            "model_selection_partition": evaluation_partition,
            "model_selection_summary": selection_summaries,
            "baseline_summary": baseline_summaries,
            "final_audit": final_audit,
            "final_audit_baseline_summary": audit_baseline_summaries,
            "best_model": best,
            "regime_summary": regime_summary,
            "failure_windows": failure_windows,
            "feature_diagnostics": feature_diagnostics,
            "oos_daily": oos_daily,
            "gate": gate,
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        report_path = output / "v8_research_report.json"
        summary_path = output / "v8_research_summary.csv"
        oos_daily_path = output / "v8_oos_daily.csv"
        oos_selection_path = output / "v8_oos_top5.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(oos_daily).to_csv(oos_daily_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(oos_selections).to_csv(oos_selection_path, index=False, encoding="utf-8-sig")
        self._write_partial(partial_path, self._partial_payload(dataset_dir, cfg, results, baseline_results, "complete"))
        self._append_progress(
            progress_path,
            {
                "event": "research_completed",
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "result_count": len(results),
            },
        )
        return {
            **payload,
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "oos_daily_path": str(oos_daily_path),
            "oos_selection_path": str(oos_selection_path),
        }

    def train_final_artifact(
        self,
        dataset_dir: str | Path,
        research_report_path: str | Path,
        output_dir: str | Path,
        config: V8ResearchConfig | None = None,
        market_gate_report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        cfg = config or V8ResearchConfig(train_windows=(240,))
        dataset = V8ParquetDataset(dataset_dir)
        dataset_audit = self.load_dataset_audit(dataset_dir, cfg)
        report = json.loads(Path(research_report_path).read_text(encoding="utf-8"))
        if report.get("report_contract") != V8_REPORT_CONTRACT:
            raise V8ResearchError(f"Unsupported v8 research report: {report.get('report_contract')}")
        if (report.get("dataset_audit") or {}).get("dataset_sha256") != dataset_audit.get("dataset_sha256"):
            raise V8ResearchError("Research report and current dataset audit hashes do not match.")
        best = report.get("best_model") or {}
        train_window = int(best.get("train_window") or max(cfg.train_windows))
        score_variant = str(best.get("score_variant") or "composite")
        if score_variant not in V8_SCORE_VARIANTS:
            raise V8ResearchError(f"Unsupported winning score variant: {score_variant}")
        selected_model = str(best.get("model") or "ensemble")
        model_names = tuple(cfg.models) if selected_model == "ensemble" else (selected_model,)
        label_probe = dataset.load_dates(dataset.dates, ["trade_date", "label_20d_end_date", "label_10d_mfe_close"])
        mature_dates = sorted(
            label_probe.loc[
                pd.to_numeric(label_probe["label_10d_mfe_close"], errors="coerce").notna()
                & label_probe["label_20d_end_date"].notna(),
                "trade_date",
            ].astype(str).unique().tolist()
        )
        train_dates = mature_dates[-train_window:]
        if not train_dates:
            raise V8ResearchError("No mature v8 dates are available for final training.")
        frame = self.clean_eligible_rows(dataset.load_dates(train_dates, self.required_columns(dataset.feature_columns)))
        distribution_sample = frame.sample(
            n=min(100_000, len(frame)),
            random_state=cfg.random_state,
        )
        training_distribution = self.feature_distribution(distribution_sample, dataset.feature_columns)
        artifact_dir = Path(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        models: dict[str, dict[str, str]] = {}
        model_sha256: dict[str, dict[str, str]] = {}
        calibrations: dict[str, dict[str, dict[str, float] | None]] = {}
        training_backends: dict[str, dict[str, str]] = {}
        for model_name in model_names:
            family: dict[str, str] = {}
            family_hashes: dict[str, str] = {}
            family_calibrations: dict[str, dict[str, float] | None] = {}
            family_backends: dict[str, str] = {}
            for head_name, spec in V8_HEADS.items():
                model = self.fit_head(frame, dataset.feature_columns, model_name, spec, cfg)
                predictions = self.predict_head(model, frame, dataset.feature_columns, spec)
                family_calibrations[head_name] = self.fit_calibration(
                    predictions,
                    pd.to_numeric(frame[spec["target"]], errors="coerce").to_numpy(),
                    spec,
                )
                path = artifact_dir / f"{model_name}_{head_name}.joblib"
                joblib.dump(model, path)
                family[head_name] = str(path)
                family_hashes[head_name] = _file_sha256(path)
                family_backends[head_name] = self.effective_backend(model_name, spec["kind"], cfg)
            models[model_name] = family
            model_sha256[model_name] = family_hashes
            calibrations[model_name] = family_calibrations
            training_backends[model_name] = family_backends
        version_seed = (
            f"{dataset.manifest.get('source_sha256')}|{train_dates[0]}|{train_dates[-1]}|"
            f"{model_names}|{score_variant}|{V8_LIVE_FEATURE_CONTRACT}"
        )
        version = f"v8_top5_{train_dates[-1]}_{sha256(version_seed.encode('utf-8')).hexdigest()[:8]}"
        market_gate = self.package_market_gate(market_gate_report_path, artifact_dir)
        metadata = {
            "artifact_contract": "ml_v8_model_artifact_v1",
            "model_version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest_path": str(dataset.manifest_path),
            "feature_set": dataset.manifest.get("feature_set") or "factor_lab_v8",
            "feature_contract": V8_LIVE_FEATURE_CONTRACT,
            "dataset_audit_path": cfg.dataset_audit_path,
            "dataset_sha256": dataset_audit.get("dataset_sha256"),
            "schema_sha256": dataset_audit.get("schema_sha256"),
            "research_report_path": str(research_report_path),
            "research_gate": report.get("gate"),
            "research_reference": report.get("final_audit"),
            "selected_research_model": selected_model,
            "score_variant": score_variant,
            "feature_columns": dataset.feature_columns,
            "heads": V8_HEADS,
            "models": models,
            "model_sha256": model_sha256,
            "training_backends": training_backends,
            "calibrations": calibrations,
            "training_distribution": training_distribution,
            "market_gate": market_gate,
            "train_dates": {"start": train_dates[0], "end": train_dates[-1], "days": len(train_dates)},
            "config": asdict(cfg),
            "diagnostic_only": report.get("gate", {}).get("status") != "candidate",
            "promotion_allowed": False,
        }
        metadata_path = artifact_dir / "artifact.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        registry = {
            "registry_contract": "ml_v8_model_registry_v1",
            "active_model_version": version,
            "entries": [
                {
                    "model_version": version,
                    "artifact_path": str(metadata_path),
                    "dataset_sha256": dataset_audit.get("dataset_sha256"),
                    "schema_sha256": dataset_audit.get("schema_sha256"),
                    "research_gate": report.get("gate"),
                    "diagnostic_only": metadata["diagnostic_only"],
                    "promotion_allowed": False,
                }
            ],
        }
        registry_path = artifact_dir / "registry.json"
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**metadata, "artifact_path": str(metadata_path), "registry_path": str(registry_path)}

    @staticmethod
    def package_market_gate(report_path: str | Path | None, artifact_dir: Path) -> dict[str, Any]:
        if not report_path:
            return {
                "status": "unavailable",
                "reason": "chronologically_audited_market_gate_report_required",
            }
        source_path = Path(report_path)
        report = json.loads(source_path.read_text(encoding="utf-8"))
        contract = report.get("report_contract")
        if contract not in {"ml_v8_market_gate_research_v1", "ml_v10_market_gate_research_v1"}:
            raise V8ResearchError(f"Unsupported v8 market-gate report: {contract}")
        source_model = report.get("model_path")
        packaged_model = None
        if source_model:
            packaged_path = artifact_dir / "market_gate.joblib"
            joblib.dump(joblib.load(source_model), packaged_path)
            packaged_model = str(packaged_path)
        gate_status = (report.get("gate") or {}).get("status")
        if contract == "ml_v10_market_gate_research_v1":
            return {
                "status": "trained" if gate_status == "candidate" else "diagnostic_observe_only",
                "research_status": gate_status,
                "report_contract": contract,
                "report_path": str(source_path),
                "model_type": report.get("selected_gate"),
                "model_path": packaged_model,
                "features": report.get("features"),
                "thresholds": report.get("thresholds"),
                "selected_stable_coverage": report.get("selected_stable_coverage"),
                "selected_aggressive_coverage": report.get("selected_aggressive_coverage"),
                "final_audit": report.get("final_audit"),
                "checks": (report.get("gate") or {}).get("checks"),
                "decision_contract": "eligible_stable/eligible_aggressive/no_signal",
                "policy": "Live decision is diagnostic unless the chronologically isolated v10 gate passes every check.",
            }
        return {
            "status": "trained" if gate_status == "diagnostic_pass" else "diagnostic_observe_only",
            "research_status": gate_status,
            "report_contract": contract,
            "report_path": str(source_path),
            "model_type": report.get("selected_gate"),
            "model_path": packaged_model,
            "features": report.get("features") or list(MARKET_STATE_FEATURES),
            "threshold": report.get("threshold"),
            "selected_target_coverage": report.get("selected_target_coverage"),
            "final_audit": report.get("final_audit"),
            "checks": (report.get("gate") or {}).get("checks"),
            "policy": "Live decision is diagnostic unless the chronologically isolated market-gate report passes every check.",
        }

    @staticmethod
    def load_dataset_audit(dataset_dir: str | Path, config: V8ResearchConfig) -> dict[str, Any]:
        if not config.dataset_audit_path:
            if config.require_dataset_audit:
                raise V8ResearchError("A passing v8 dataset audit is required.")
            return {"status": "missing", "reason": "dataset_audit_path_not_configured"}
        try:
            report = V8DatasetAuditor.validate_report(config.dataset_audit_path, dataset_dir)
        except V8DatasetAuditError as exc:
            if config.require_dataset_audit:
                raise V8ResearchError(str(exc)) from exc
            return {"status": "blocked", "reason": str(exc)}
        return {
            "status": "pass",
            "report_path": str(config.dataset_audit_path),
            "dataset_sha256": report.get("dataset_sha256"),
            "schema_sha256": report.get("schema_sha256"),
            "source_sha256": report.get("source_sha256"),
            "sample_rows": (report.get("label_reconstruction") or {}).get("sample_rows"),
            "reconstruction_mismatch_count": (report.get("label_reconstruction") or {}).get("mismatch_count"),
        }

    @staticmethod
    def feature_distribution(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float | None]]:
        output: dict[str, dict[str, float | None]] = {}
        for feature in feature_columns:
            values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            clean = values.dropna()
            output[feature] = {
                "median": _safe_number(clean.median()) if not clean.empty else None,
                "q25": _safe_number(clean.quantile(0.25)) if not clean.empty else None,
                "q75": _safe_number(clean.quantile(0.75)) if not clean.empty else None,
                "missing_rate": round(float(values.isna().mean()), 6),
            }
        return output

    @staticmethod
    def fit_calibration(predictions: np.ndarray, targets: np.ndarray, spec: dict[str, Any]) -> dict[str, float] | None:
        if spec["kind"] == "classification":
            return None
        usable = np.isfinite(predictions) & np.isfinite(targets)
        if usable.sum() < 10 or float(np.std(predictions[usable])) == 0.0:
            return {"slope": 0.0, "intercept": float(np.nanmean(targets[usable])) if usable.any() else 0.0}
        slope, intercept = np.polyfit(predictions[usable], targets[usable], 1)
        return {"slope": float(slope), "intercept": float(intercept)}

    @staticmethod
    def train_market_gate(oos_daily: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
        daily = pd.DataFrame(oos_daily)
        if len(daily) < 60:
            return {"status": "unavailable", "reason": "fewer_than_60_oos_days"}
        daily["selection_positive"] = (
            (pd.to_numeric(daily["mean_net_return_5d"], errors="coerce") > 0)
            & daily["hit5pct_10d"].astype(str).str.lower().isin({"true", "1"})
        ).astype(int)
        if daily["selection_positive"].nunique() < 2:
            return {"status": "unavailable", "reason": "single_class_oos_target"}
        try:
            state, provenance = MarketStateResearch.attach_raw_market_state(daily, factor_set="factor_lab_v2")
        except Exception as exc:
            return {"status": "unavailable", "reason": f"raw_market_state_error:{exc}"}
        from lightgbm import LGBMClassifier

        features = list(MARKET_STATE_FEATURES)
        x = state[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model = LGBMClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, num_leaves=7, min_child_samples=20, random_state=42, verbose=-1)
        model.fit(x, state["selection_positive"])
        probabilities = model.predict_proba(x)[:, 1]
        threshold = float(pd.Series(probabilities).quantile(0.75))
        path = artifact_dir / "market_gate.joblib"
        joblib.dump(model, path)
        return {
            "status": "trained",
            "model_path": str(path),
            "features": features,
            "threshold": threshold,
            "training_days": int(len(state)),
            "target_positive_rate": round(float(state["selection_positive"].mean()), 6),
            "target_coverage": 0.25,
            "state_provenance": provenance,
            "policy": "trained only from ranker OOS daily outcomes; live features are signal-date raw Tushare market state",
        }

    @staticmethod
    def required_columns(feature_columns: list[str]) -> list[str]:
        return sorted(
            set(
                feature_columns
                + [
                    "trade_date",
                    "ts_code",
                    "industry",
                    "stable_score",
                    "aggressive_score",
                    "label_entry_tradable",
                    "label_20d_end_date",
                    "label_5d_net_return",
                    "label_10d_mfe_close",
                    "label_10d_mae_close",
                    "label_10d_hit_5pct",
                    "label_20d_hit_8pct",
                ]
            )
        )

    @staticmethod
    def walk_forward_windows(dates: list[str], train_days: int, config: V8ResearchConfig) -> list[dict[str, Any]]:
        windows = []
        for offset in range(0, len(dates) - train_days - config.validation_days + 1, config.step_days):
            windows.append(
                {
                    "train_dates": dates[offset : offset + train_days],
                    "validation_dates": dates[offset + train_days : offset + train_days + config.validation_days],
                }
            )
        return windows

    @staticmethod
    def apply_embargo(train: pd.DataFrame, validation_start: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        end_dates = train["label_20d_end_date"].astype(str).str.replace(".0", "", regex=False)
        keep = end_dates < str(validation_start)
        output = train[keep].copy()
        overlap = int((output["label_20d_end_date"].astype(str) >= str(validation_start)).sum())
        return output, {
            "status": "pass" if overlap == 0 else "blocked",
            "rows_before": int(len(train)),
            "rows_after": int(len(output)),
            "embargoed_rows": int((~keep).sum()),
            "overlap_after": overlap,
            "policy": "label_20d_end_date must precede validation_start",
        }

    @staticmethod
    def clean_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
        tradable = frame["label_entry_tradable"].astype(str).str.lower().isin({"true", "1"})
        required_labels = [spec["target"] for spec in V8_HEADS.values()]
        valid = tradable.copy()
        for column in required_labels:
            valid &= pd.to_numeric(frame[column], errors="coerce").notna()
        return frame[valid].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def evaluate_model_family(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        feature_columns: list[str],
        model_name: str,
        config: V8ResearchConfig,
        *,
        collect_permutation: bool = False,
    ) -> dict[str, Any]:
        valid_scores: dict[str, np.ndarray] = {}
        training_backends: dict[str, str] = {}
        fitted_models: dict[str, Any] = {}
        head_importances: dict[str, np.ndarray] = {}
        for head_name, spec in V8_HEADS.items():
            model = self.fit_head(train, feature_columns, model_name, spec, config)
            fitted_models[head_name] = model
            training_backends[head_name] = self.effective_backend(model_name, spec["kind"], config)
            valid_scores[head_name] = self.predict_head(model, valid, feature_columns, spec)
            importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), dtype=float)
            if len(importance) == len(feature_columns):
                head_importances[head_name] = importance
        variant_frames = self.score_variants(valid, valid_scores)
        variant_metrics = {
            name: evaluate_top5(frame, selection_mode=self.selection_mode_for_variant(name))
            for name, frame in variant_frames.items()
        }
        builtin_importance = self.aggregate_feature_importance(feature_columns, head_importances)
        permutation = []
        if collect_permutation and builtin_importance:
            permutation = self.permutation_diagnostics(
                fitted_models,
                valid,
                feature_columns,
                builtin_importance[: config.permutation_feature_count],
                variant_metrics,
                config,
            )
        variants = {
            name: {
                "metrics": _compact_metrics(metrics),
                "permutation_diagnostics": [row for row in permutation if row["score_variant"] == name],
                "oos_daily": metrics.get("daily", []),
                "oos_selections": metrics.get("selections", []),
            }
            for name, metrics in variant_metrics.items()
        }
        composite = variants["composite"]
        return {
            "status": "trained",
            "model": model_name,
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "training_backends": training_backends,
            "metrics": composite["metrics"],
            "feature_importance": builtin_importance[:50],
            "permutation_diagnostics": composite["permutation_diagnostics"],
            "oos_daily": composite["oos_daily"],
            "oos_selections": composite["oos_selections"],
            "variants": variants,
            "_variant_predictions": {
                name: frame["prediction"].to_numpy(dtype=float) for name, frame in variant_frames.items()
            },
        }

    def evaluate_model_ensemble(
        self,
        valid: pd.DataFrame,
        family_predictions: dict[str, dict[str, np.ndarray]],
        family_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dates = valid["trade_date"].astype(str)
        variants: dict[str, dict[str, Any]] = {}
        for variant in V8_SCORE_VARIANTS:
            ranked = []
            for predictions in family_predictions.values():
                values = pd.Series(predictions[variant], index=valid.index)
                ranked.append(values.groupby(dates, sort=False).rank(pct=True).to_numpy(dtype=float))
            matrix = np.vstack(ranked)
            scored = valid.copy()
            scored["prediction"] = matrix.mean(axis=0)
            scored["model_disagreement"] = matrix.std(axis=0)
            if variant == V11_DUAL_VARIANT:
                scored["v11_stable_score"], scored["v11_aggressive_score"] = self.ensemble_v11_scores(
                    valid,
                    family_predictions,
                )
            metrics = evaluate_top5(scored, selection_mode=self.selection_mode_for_variant(variant))
            variants[variant] = {
                "metrics": _compact_metrics(metrics),
                "permutation_diagnostics": self._ensemble_permutation_rows(family_results, variant),
                "oos_daily": metrics.get("daily", []),
                "oos_selections": metrics.get("selections", []),
            }
        feature_importance = self._mean_feature_importance(family_results)
        composite = variants["composite"]
        return {
            "status": "trained",
            "model": "ensemble",
            "train_rows": family_results[0].get("train_rows") if family_results else 0,
            "validation_rows": int(len(valid)),
            "training_backends": {
                row["model"]: row.get("training_backends") for row in family_results
            },
            "metrics": composite["metrics"],
            "feature_importance": feature_importance,
            "permutation_diagnostics": composite["permutation_diagnostics"],
            "oos_daily": composite["oos_daily"],
            "oos_selections": composite["oos_selections"],
            "variants": variants,
        }

    @staticmethod
    def _ensemble_permutation_rows(family_results: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
        rows = []
        for result in family_results:
            rows.extend((result.get("variants") or {}).get(variant, {}).get("permutation_diagnostics") or [])
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        return [
            {
                "score_variant": variant,
                "feature": feature,
                "base_importance": _safe_number(group["base_importance"].mean()),
                "delta_mean_net_return_at5": _safe_number(group["delta_mean_net_return_at5"].mean()),
                "delta_ndcg_at5": _safe_number(group["delta_ndcg_at5"].mean()),
                "method": "mean_family_within_trade_date_permutation",
            }
            for feature, group in frame.groupby("feature", sort=False)
        ]

    @staticmethod
    def _mean_feature_importance(family_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        values = Counter()
        counts = Counter()
        for result in family_results:
            for item in result.get("feature_importance") or []:
                values[item["feature"]] += float(item["importance"])
                counts[item["feature"]] += 1
        return sorted(
            (
                {"feature": feature, "importance": round(total / counts[feature], 8)}
                for feature, total in values.items()
            ),
            key=lambda item: item["importance"],
            reverse=True,
        )[:50]

    def fit_head(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        model_name: str,
        spec: dict[str, Any],
        config: V8ResearchConfig,
    ) -> Any:
        x = self.clean_features(frame, feature_columns)
        target = pd.to_numeric(frame[spec["target"]], errors="coerce")
        if spec["kind"] == "rank":
            target = target.groupby(frame["trade_date"].astype(str), sort=False).rank(pct=True).mul(30).round().astype(np.int16)
        model = self.make_model(model_name, spec["kind"], config)
        if spec["kind"] == "rank":
            group_sizes = frame.groupby("trade_date", sort=False).size().tolist()
            if model_name == "lightgbm":
                model.fit(x, target, group=group_sizes)
            elif model_name == "xgboost":
                qid = pd.factorize(frame["trade_date"].astype(str), sort=False)[0]
                model.fit(x, target, qid=qid)
            else:
                model.fit(x, target, group_id=frame["trade_date"].astype(str).to_numpy())
        else:
            model.fit(x, target)
        return model

    @staticmethod
    def make_model(model_name: str, kind: str, config: V8ResearchConfig) -> Any:
        gpu = config.device == "gpu"
        common = {"random_state": config.random_state, "n_estimators": config.n_estimators, "learning_rate": 0.04, "max_depth": 5}
        if model_name == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

            extra = {**common, "num_leaves": 31, "verbose": -1, "device_type": "gpu" if gpu else "cpu"}
            if kind == "rank":
                return LGBMRanker(objective="lambdarank", **extra)
            if kind == "classification":
                return LGBMClassifier(**extra)
            return LGBMRegressor(**extra)
        if model_name == "xgboost":
            from xgboost import XGBClassifier, XGBRanker, XGBRegressor

            extra = {**common, "tree_method": "hist", "device": "cuda" if gpu else "cpu", "subsample": 0.85, "colsample_bytree": 0.85}
            if kind == "rank":
                return XGBRanker(objective="rank:ndcg", **extra)
            if kind == "classification":
                return XGBClassifier(eval_metric="logloss", **extra)
            return XGBRegressor(**extra)
        if model_name == "catboost":
            from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

            # CatBoost CUDA querywise losses reject full A-share daily groups above
            # 1023 rows. Keep one complete market-day query and use CPU for rank heads.
            effective_gpu = gpu and kind != "rank"
            extra = {
                "iterations": config.n_estimators,
                "learning_rate": 0.04,
                "depth": 5,
                "random_seed": config.random_state,
                "verbose": False,
                "allow_writing_files": False,
                "task_type": "GPU" if effective_gpu else "CPU",
            }
            if effective_gpu:
                extra["devices"] = "0"
            if kind == "rank":
                return CatBoostRanker(loss_function="YetiRank", **extra)
            if kind == "classification":
                return CatBoostClassifier(**extra)
            return CatBoostRegressor(**extra)
        raise V8ResearchError(f"Unsupported v8 model: {model_name}")

    @staticmethod
    def effective_backend(model_name: str, kind: str, config: V8ResearchConfig) -> str:
        if config.device != "gpu":
            return "cpu"
        if model_name == "catboost" and kind == "rank":
            return "cpu_full_day_group_catboost_cuda_query_limit"
        return "gpu"

    @staticmethod
    def clean_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        return frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    @staticmethod
    def predict_head(model: Any, frame: pd.DataFrame, feature_columns: list[str], spec: dict[str, Any]) -> np.ndarray:
        x = V8ResearchHarness.clean_features(frame, feature_columns)
        if model.__class__.__module__.startswith("xgboost") and hasattr(model, "get_booster"):
            model.set_params(device="cpu")
            model.get_booster().set_param({"device": "cpu"})
        if spec["kind"] == "classification" and hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(x))[:, -1]
        return np.asarray(model.predict(x)).reshape(-1)

    @staticmethod
    def aggregate_feature_importance(
        feature_columns: list[str],
        head_importances: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        if not head_importances:
            return []
        normalized = []
        for values in head_importances.values():
            total = float(np.nansum(np.abs(values)))
            normalized.append(np.abs(values) / total if total > 0 else np.zeros_like(values))
        mean = np.vstack(normalized).mean(axis=0)
        order = np.argsort(-mean)
        return [
            {"feature": feature_columns[index], "importance": round(float(mean[index]), 8)}
            for index in order
        ]

    def permutation_diagnostics(
        self,
        models: dict[str, Any],
        valid: pd.DataFrame,
        feature_columns: list[str],
        top_features: list[dict[str, Any]],
        base_metrics: dict[str, dict[str, Any]],
        config: V8ResearchConfig,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(config.random_state)
        output = []
        dates = valid["trade_date"].astype(str)
        for item in top_features:
            feature = item["feature"]
            permuted = valid.copy()
            values = permuted[feature].copy()
            permuted[feature] = values.groupby(dates, sort=False).transform(
                lambda group: rng.permutation(group.to_numpy())
            )
            scores = {
                name: self.predict_head(models[name], permuted, feature_columns, spec)
                for name, spec in V8_HEADS.items()
            }
            for variant, permuted_scored in self.score_variants(permuted, scores).items():
                metrics = evaluate_top5(permuted_scored, selection_mode=self.selection_mode_for_variant(variant))
                reference = base_metrics[variant]
                output.append(
                    {
                        "score_variant": variant,
                        "feature": feature,
                        "base_importance": item["importance"],
                        "delta_mean_net_return_at5": _metric_delta(
                            metrics.get("mean_net_return_at5"), reference.get("mean_net_return_at5")
                        ),
                        "delta_ndcg_at5": _metric_delta(metrics.get("ndcg_at5"), reference.get("ndcg_at5")),
                        "method": "within_trade_date_permutation_on_untouched_oos_fold",
                    }
                )
        return output

    @classmethod
    def score_variants(cls, frame: pd.DataFrame, head_scores: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
        composite = cls.composite_scores(frame.copy(), head_scores)
        output = {"composite": composite}
        for head_name, values in head_scores.items():
            scored = frame.copy()
            scored["prediction"] = values
            scored["model_disagreement"] = composite["model_disagreement"].to_numpy()
            output[head_name] = scored
        v11 = composite.copy()
        v11["v11_stable_score"], v11["v11_aggressive_score"] = cls.v11_dual_scores(composite)
        v11["prediction"] = (v11["v11_stable_score"] + v11["v11_aggressive_score"]) / 2.0
        output[V11_DUAL_VARIANT] = v11
        return output

    @staticmethod
    def composite_scores(frame: pd.DataFrame, head_scores: dict[str, np.ndarray]) -> pd.DataFrame:
        component_columns = []
        for head_name, values in head_scores.items():
            column = f"prediction_{head_name}"
            frame[column] = values
            normalized = pd.Series(values, index=frame.index).groupby(frame["trade_date"].astype(str), sort=False).rank(pct=True)
            frame[f"normalized_{head_name}"] = normalized
            component_columns.append((head_name, normalized))
        frame["prediction"] = sum(V8_HEADS[name]["weight"] * values for name, values in component_columns)
        frame["model_disagreement"] = frame[[f"normalized_{name}" for name in V8_HEADS]].std(axis=1, ddof=0)
        return frame

    @staticmethod
    def v11_dual_scores(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        disagreement_bonus = 1.0 - pd.to_numeric(frame.get("model_disagreement", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        return_5d = pd.to_numeric(frame["normalized_return_5d"], errors="coerce").fillna(0.0)
        hit5 = pd.to_numeric(frame["normalized_hit5_10d"], errors="coerce").fillna(0.0)
        upside = pd.to_numeric(frame["normalized_upside_10d"], errors="coerce").fillna(0.0)
        safety = pd.to_numeric(frame["normalized_safety_10d"], errors="coerce").fillna(0.0)
        stable = 0.45 * return_5d + 0.35 * safety + 0.15 * hit5 + 0.05 * disagreement_bonus
        aggressive = 0.45 * upside + 0.35 * hit5 + 0.15 * return_5d + 0.05 * safety
        return stable, aggressive

    @staticmethod
    def ensemble_v11_scores(
        valid: pd.DataFrame,
        family_predictions: dict[str, dict[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        dates = valid["trade_date"].astype(str)
        stable_parts = []
        aggressive_parts = []
        for predictions in family_predictions.values():
            ranked = {
                name: pd.Series(predictions[name], index=valid.index).groupby(dates, sort=False).rank(pct=True)
                for name in ("return_5d", "hit5_10d", "upside_10d", "safety_10d")
            }
            stable_parts.append(
                (0.45 * ranked["return_5d"] + 0.35 * ranked["safety_10d"] + 0.15 * ranked["hit5_10d"]).to_numpy(dtype=float)
            )
            aggressive_parts.append(
                (0.45 * ranked["upside_10d"] + 0.35 * ranked["hit5_10d"] + 0.15 * ranked["return_5d"] + 0.05 * ranked["safety_10d"]).to_numpy(dtype=float)
            )
        return np.vstack(stable_parts).mean(axis=0), np.vstack(aggressive_parts).mean(axis=0)

    @staticmethod
    def selection_mode_for_variant(variant: str) -> str:
        return "v11_dual_70_30" if variant == V11_DUAL_VARIANT else "diversified"

    @staticmethod
    def baseline_predictions(frame: pd.DataFrame, random_state: int) -> dict[str, pd.DataFrame]:
        outputs: dict[str, pd.DataFrame] = {}
        for name, column in (
            ("rule_stable", "stable_score"),
            ("rule_aggressive", "aggressive_score"),
            ("simple_return_20d", "feature_return_20d"),
            ("simple_flow_5d", "feature_main_net_to_amount_sum_5d"),
        ):
            if column in frame.columns:
                scored = frame.copy()
                scored["prediction"] = pd.to_numeric(scored[column], errors="coerce")
                outputs[name] = scored
        random = frame.copy()
        random["prediction"] = np.random.default_rng(random_state).random(len(random))
        outputs["random"] = random
        return outputs

    @staticmethod
    def summarize_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trained = [row for row in results if row.get("status") == "trained"]
        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in trained:
            groups.setdefault((row["model"], row["train_window"], row.get("score_variant") or "composite"), []).append(row)
        return [
            {"model": model, "train_window": train_window, "score_variant": variant, **summarize_folds(rows)}
            for (model, train_window, variant), rows in sorted(groups.items())
        ]

    @staticmethod
    def _append_progress(path: Path, event: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _write_partial(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temp_path.replace(path)

    @classmethod
    def _partial_payload(
        cls,
        dataset_dir: str | Path,
        config: V8ResearchConfig,
        results: list[dict[str, Any]],
        baseline_results: dict[str, list[dict[str, Any]]],
        status: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(dataset_dir),
            "config": asdict(config),
            "result_count": len(results),
            "trained_result_count": sum(1 for row in results if row.get("status") == "trained"),
            "baseline_result_count": sum(len(rows) for rows in baseline_results.values()),
            "summary": cls.summarize_models(cls.flatten_variant_results(results)),
            "baseline_summary": [{
                "baseline": name,
                **summarize_folds(rows),
            } for name, rows in sorted(baseline_results.items())],
            "results": cls.compact_results_for_report(results),
        }

    @staticmethod
    def flatten_variant_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in results:
            if row.get("status") != "trained":
                continue
            common = {
                key: value
                for key, value in row.items()
                if key not in {"variants", "metrics", "permutation_diagnostics", "oos_daily", "oos_selections"}
            }
            for variant, details in (row.get("variants") or {}).items():
                output.append(
                    {
                        **common,
                        "score_variant": variant,
                        "metrics": details.get("metrics") or {},
                        "permutation_diagnostics": details.get("permutation_diagnostics") or [],
                        "oos_daily": details.get("oos_daily") or [],
                        "oos_selections": details.get("oos_selections") or [],
                    }
                )
        return output

    @staticmethod
    def compact_results_for_report(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for row in results:
            item = {
                key: value
                for key, value in row.items()
                if key not in {"oos_daily", "oos_selections"}
            }
            item["variants"] = {
                variant: {
                    "metrics": details.get("metrics"),
                    "permutation_diagnostics": details.get("permutation_diagnostics") or [],
                }
                for variant, details in (row.get("variants") or {}).items()
            }
            compact.append(item)
        return compact

    @staticmethod
    def best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not summaries:
            return None
        return max(
            summaries,
            key=lambda row: (
                float(row.get("mean_net_return_at5") or -999),
                float(row.get("hit5pct_10d_at5") or 0),
                -float(row.get("all_loss_rate_at5") or 1),
            ),
        )

    @staticmethod
    def best_oos_stream(
        results: list[dict[str, Any]],
        best: dict[str, Any] | None,
        allowed_validation_starts: set[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not best:
            return [], []
        rows = [
            row for row in results
            if row.get("status") == "trained" and row.get("model") == best.get("model") and row.get("train_window") == best.get("train_window")
            and row.get("score_variant") == best.get("score_variant")
            and (allowed_validation_starts is None or row.get("validation_start") in allowed_validation_starts)
        ]
        daily = [{**item, "fold_index": row["fold_index"]} for row in rows for item in row.get("oos_daily", [])]
        selections = [{**item, "fold_index": row["fold_index"]} for row in rows for item in row.get("oos_selections", [])]
        return daily, selections

    @staticmethod
    def partition_model_selection(results: list[dict[str, Any]], config: V8ResearchConfig) -> dict[str, Any]:
        trained = [row for row in results if row.get("status") == "trained"]
        configurations = sorted({(row["model"], row["train_window"], row.get("score_variant") or "composite") for row in trained})
        date_configurations: dict[str, set[tuple[str, int, str]]] = {}
        for row in trained:
            date_configurations.setdefault(row["validation_start"], set()).add(
                (row["model"], row["train_window"], row.get("score_variant") or "composite")
            )
        common = sorted(date for date, values in date_configurations.items() if set(configurations).issubset(values))
        audit_count = max(config.min_final_audit_folds, int(np.ceil(len(common) * config.final_audit_fraction)))
        if len(common) <= audit_count:
            return {
                "status": "blocked",
                "reason": "insufficient_common_folds_for_separate_model_selection_and_final_audit",
                "configuration_count": len(configurations),
                "common_fold_count": len(common),
                "selection_validation_starts": [],
                "audit_validation_starts": common,
                "policy": "Model selection and final gate evaluation must use chronologically separate OOS folds.",
            }
        return {
            "status": "pass",
            "configuration_count": len(configurations),
            "common_fold_count": len(common),
            "selection_fold_count": len(common) - audit_count,
            "audit_fold_count": audit_count,
            "selection_validation_starts": common[:-audit_count],
            "audit_validation_starts": common[-audit_count:],
            "policy": "Earlier common OOS folds select the model family/window; the latest common OOS folds are untouched final audit.",
        }

    @classmethod
    def planned_validation_partition(cls, dates: list[str], config: V8ResearchConfig) -> dict[str, Any]:
        starts_by_window = []
        for train_window in config.train_windows:
            folds = cls.walk_forward_windows(dates, train_window, config)
            if config.max_folds is not None:
                folds = folds[: config.max_folds]
            starts_by_window.append({fold["validation_dates"][0] for fold in folds})
        common = sorted(set.intersection(*starts_by_window)) if starts_by_window else []
        audit_count = max(config.min_final_audit_folds, int(np.ceil(len(common) * config.final_audit_fraction)))
        return {
            "selection_validation_starts": common[:-audit_count] if len(common) > audit_count else [],
            "audit_validation_starts": common[-audit_count:] if len(common) > audit_count else [],
        }

    @staticmethod
    def feature_diagnostics(audit_results: list[dict[str, Any]]) -> dict[str, Any]:
        importance = Counter()
        counts = Counter()
        permutation_rows: list[dict[str, Any]] = []
        for row in audit_results:
            for item in row.get("feature_importance") or []:
                importance[item["feature"]] += float(item["importance"])
                counts[item["feature"]] += 1
            permutation_rows.extend(row.get("permutation_diagnostics") or [])
        built_in = sorted(
            (
                {"feature": feature, "mean_importance": round(value / counts[feature], 8)}
                for feature, value in importance.items()
            ),
            key=lambda item: item["mean_importance"],
            reverse=True,
        )
        permutation_frame = pd.DataFrame(permutation_rows)
        permutation = []
        if not permutation_frame.empty:
            for feature, rows in permutation_frame.groupby("feature", sort=False):
                permutation.append(
                    {
                        "feature": feature,
                        "folds": int(len(rows)),
                        "mean_delta_mean_net_return_at5": _safe_number(rows["delta_mean_net_return_at5"].mean()),
                        "mean_delta_ndcg_at5": _safe_number(rows["delta_ndcg_at5"].mean()),
                    }
                )
        return {
            "builtin_importance": built_in[:50],
            "permutation_diagnostics": sorted(
                permutation,
                key=lambda item: float(item.get("mean_delta_mean_net_return_at5") or 0.0),
            ),
            "permutation_policy": "Only untouched final-audit folds; values are shuffled within each trade date.",
        }

    @staticmethod
    def failure_windows(oos_daily: list[dict[str, Any]], audit_results: list[dict[str, Any]]) -> dict[str, Any]:
        daily = pd.DataFrame(oos_daily)
        worst_days: list[dict[str, Any]] = []
        if not daily.empty:
            ranked = daily.assign(
                _return=pd.to_numeric(daily["mean_net_return_5d"], errors="coerce")
            ).sort_values("_return", ascending=True).head(20)
            worst_days = [
                {
                    "trade_date": str(row["trade_date"]),
                    "mean_net_return_at5": _safe_number(row.get("mean_net_return_5d")),
                    "all_loss_at5": bool(row.get("all_loss_5d")),
                    "market_mean_net_return_5d": _safe_number(row.get("market_mean_net_return_5d")),
                }
                for row in ranked.to_dict("records")
            ]
        negative_folds = [
            {
                "fold_index": row.get("fold_index"),
                "validation_start": row.get("validation_start"),
                "validation_end": row.get("validation_end"),
                "mean_net_return_at5": (row.get("metrics") or {}).get("mean_net_return_at5"),
                "all_loss_rate_at5": (row.get("metrics") or {}).get("all_loss_rate_at5"),
            }
            for row in audit_results
            if float((row.get("metrics") or {}).get("mean_net_return_at5") or 0.0) <= 0.0
        ]
        return {"worst_days": worst_days, "negative_folds": negative_folds}

    @staticmethod
    def regime_audit(oos_daily: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        frame = pd.DataFrame(oos_daily)
        if frame.empty:
            return [], False
        market = pd.to_numeric(frame["market_mean_net_return_5d"], errors="coerce")
        frame["regime"] = np.where(market >= 0.005, "broad_up", np.where(market <= -0.005, "broad_down", "mixed"))
        summary = []
        for regime, rows in frame.groupby("regime"):
            summary.append({"regime": regime, "days": int(len(rows)), "mean_net_return_at5": round(float(pd.to_numeric(rows["mean_net_return_5d"], errors="coerce").mean()), 6)})
        material = [row for row in summary if row["days"] >= 10]
        passed = len(material) >= 2 and all(float(row["mean_net_return_at5"]) > -0.005 for row in material)
        return summary, passed


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"daily", "selections"}}


def _safe_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 6) if np.isfinite(result) else None


def _metric_delta(value: Any, reference: Any) -> float | None:
    left = _safe_number(value)
    right = _safe_number(reference)
    return round(left - right, 6) if left is not None and right is not None else None


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()
