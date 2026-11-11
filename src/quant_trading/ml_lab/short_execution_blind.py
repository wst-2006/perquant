"""Frozen forward blind test for the short-execution research contract."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .short_execution import SHORT_EXECUTION_CONTRACT, compute_short_execution_labels_for_panel
from .short_execution_research import (
    SHORT_HEADS,
    ShortExecutionDataset,
    ShortExecutionResearchConfig,
    ShortExecutionResearchHarness,
    evaluate_short_top5,
)
from .v8_features import build_live_v8_feature_frame
from .v8_labels import V8MarketPanel
from .v9_events import V9_EVENT_FEATURES


BLIND_CONTRACT = "short_execution_frozen_blind_v1"


class ShortExecutionBlindError(RuntimeError):
    pass


class ShortExecutionBlindRunner:
    def run(
        self,
        *,
        historical_feature_dir: str | Path,
        historical_label_dir: str | Path,
        event_snapshot_dir: str | Path,
        output_dir: str | Path,
        train_days: int = 100,
        n_estimators: int = 120,
        random_state: int = 42,
    ) -> dict[str, Any]:
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()):
            raise ShortExecutionBlindError(f"Output directory must be empty or absent: {output}")
        output.mkdir(parents=True, exist_ok=True)
        historical = ShortExecutionDataset(historical_feature_dir, historical_label_dir)
        event_root = Path(event_snapshot_dir)
        event_manifest_path = event_root / "manifest.json"
        if not event_manifest_path.exists():
            raise ShortExecutionBlindError(f"Missing event snapshot manifest: {event_manifest_path}")
        event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
        if event_manifest.get("contract") != "ml_v9_daily_event_snapshot_v1":
            raise ShortExecutionBlindError("Unsupported v9 event snapshot contract")
        blind_dates = sorted((event_manifest.get("rows_by_date") or {}).keys())
        if not blind_dates:
            raise ShortExecutionBlindError("Event snapshot has no blind dates")
        if max(historical.dates) >= min(blind_dates):
            raise ShortExecutionBlindError("Historical and blind signal periods overlap")
        train_dates = [date for date in historical.dates if date < min(blind_dates)][-train_days:]
        if len(train_dates) != train_days:
            raise ShortExecutionBlindError(f"Need {train_days} historical training dates, found {len(train_dates)}")

        feature_dir = output / "features"
        label_dir = output / "labels"
        model_dir = output / "models"
        feature_dir.mkdir()
        label_dir.mkdir()
        model_dir.mkdir()
        feature_audit = self._build_features(historical.feature_columns, event_root, blind_dates, feature_dir)
        label_audit = self._build_labels(feature_dir, blind_dates, label_dir)

        train = historical.load_dates(train_dates)
        train, embargo = ShortExecutionResearchHarness.apply_embargo(train, min(blind_dates))
        train = ShortExecutionResearchHarness.clean_rows(train)
        blind = self._load_blind(feature_dir, label_dir, blind_dates)
        score_eligible = blind["base_factor_snapshot_available"].fillna(False).astype(bool)
        settled = blind["short_settled"].fillna(False).astype(bool)
        realized = pd.to_numeric(blind["short_realized_net_return"], errors="coerce").notna()
        scored_input = blind[score_eligible & settled & realized].copy().reset_index(drop=True)
        if scored_input.groupby("trade_date").size().min() < 5:
            raise ShortExecutionBlindError("A blind date has fewer than five settled, score-eligible stocks")

        config = ShortExecutionResearchConfig(
            train_windows=(train_days,), models=("lightgbm",), device="cpu",
            n_estimators=n_estimators, random_state=random_state,
        )
        harness = ShortExecutionResearchHarness()
        distribution_audit = self._distribution_audit(train, blind, historical.feature_columns)
        scores: dict[str, np.ndarray] = {}
        model_hashes: dict[str, str] = {}
        for name, spec in SHORT_HEADS.items():
            model = harness.models.fit_head(train, historical.feature_columns, "lightgbm", spec, config.v8_config())
            scores[name] = harness.models.predict_head(model, scored_input, historical.feature_columns, spec)
            model_path = model_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            model_hashes[name] = _file_sha256(model_path)
        scored = harness.composite(scored_input, scores)
        evaluation = evaluate_short_top5(scored)

        random_scored = scored_input.copy()
        random_scored["prediction"] = np.random.default_rng(random_state).random(len(random_scored))
        random_scored["model_disagreement"] = 0.0
        momentum = scored_input.copy()
        momentum["prediction"] = pd.to_numeric(momentum["feature_return_20d"], errors="coerce")
        momentum["model_disagreement"] = 0.0
        baselines = {
            "random_top5_fixed_seed": evaluate_short_top5(random_scored)["summary"],
            "simple_return_20d_top5": evaluate_short_top5(momentum)["summary"],
        }
        top5_path = output / "blind_top5.csv"
        daily_path = output / "blind_daily.csv"
        pd.DataFrame(evaluation["selections"]).to_csv(top5_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(evaluation["daily"]).to_csv(daily_path, index=False, encoding="utf-8-sig")
        report = {
            "task": "short_execution_frozen_forward_blind",
            "contract": BLIND_CONTRACT,
            "execution_contract": SHORT_EXECUTION_CONTRACT,
            "historical_feature_dir": str(historical_feature_dir),
            "historical_label_dir": str(historical_label_dir),
            "event_snapshot_dir": str(event_snapshot_dir),
            "feature_count": len(historical.feature_columns),
            "feature_columns": historical.feature_columns,
            "heads": SHORT_HEADS,
            "config": asdict(config),
            "split": {
                "train_signal_dates": {"start": train_dates[0], "end": train_dates[-1], "count": len(train_dates)},
                "blind_signal_dates": {"start": blind_dates[0], "end": blind_dates[-1], "count": len(blind_dates)},
                "policy": "models are fitted once before the first blind signal date and never refitted inside the blind period",
                "status": "pass",
            },
            "embargo": embargo,
            "train_rows": int(len(train)),
            "blind_universe_rows": int(len(blind)),
            "blind_score_eligible_rows": int(score_eligible.sum()),
            "blind_data_quality_excluded_rows": int((~score_eligible).sum()),
            "feature_snapshot_audit": feature_audit,
            "label_snapshot_audit": label_audit,
            "feature_distribution_audit": distribution_audit,
            "model_sha256": model_hashes,
            "blind_evaluation": evaluation["summary"],
            "baselines": baselines,
            "top5_path": str(top5_path),
            "daily_path": str(daily_path),
            "diagnostic_only": True,
            "promotion_allowed": False,
            "status": "observe_only",
        }
        report_path = output / "blind_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**report, "report_path": str(report_path)}

    @staticmethod
    def _build_features(feature_columns: list[str], event_root: Path, dates: list[str], output: Path) -> dict[str, Any]:
        v9_columns = [name for name in feature_columns if name in V9_EVENT_FEATURES]
        base_columns = [name for name in feature_columns if name not in V9_EVENT_FEATURES]
        rows = 0
        excluded = 0
        hashes: dict[str, str] = {}
        for date in dates:
            events = pd.read_parquet(event_root / f"trade_date={date}" / "part-000.parquet")
            base = build_live_v8_feature_frame(date, base_columns)
            base = pd.concat(
                [base.copy(), pd.Series(True, index=base.index, name="base_factor_snapshot_available")],
                axis=1,
            )
            keep = ["trade_date", "ts_code", "industry", "base_factor_snapshot_available", *base_columns]
            frame = events.merge(base[keep], on=["trade_date", "ts_code"], how="left", validate="one_to_one")
            frame["base_factor_snapshot_available"] = frame["base_factor_snapshot_available"].fillna(False).astype(bool)
            excluded += int((~frame["base_factor_snapshot_available"]).sum())
            if frame.duplicated(["trade_date", "ts_code"]).any() or len(frame) != len(events):
                raise ShortExecutionBlindError(f"Full-market feature merge failed for {date}")
            missing = sorted(set(feature_columns) - set(frame.columns))
            if missing:
                raise ShortExecutionBlindError(f"Blind frame missing features on {date}: {missing}")
            path = output / f"trade_date={date}" / "part-000.parquet"
            path.parent.mkdir(parents=True)
            frame[["trade_date", "ts_code", "industry", "base_factor_snapshot_available", *feature_columns]].to_parquet(
                path, index=False, engine="pyarrow", compression="zstd"
            )
            rows += len(frame)
            hashes[date] = _file_sha256(path)
        return {"status": "pass", "rows": rows, "data_quality_excluded_rows": excluded, "partition_sha256": hashes}

    @staticmethod
    def _build_labels(feature_dir: Path, dates: list[str], output: Path) -> dict[str, Any]:
        panel = V8MarketPanel.from_database(start_date=dates[0])
        settled = 0
        rows = 0
        hashes: dict[str, str] = {}
        for date in dates:
            keys = pd.read_parquet(feature_dir / f"trade_date={date}" / "part-000.parquet", columns=["trade_date", "ts_code"])
            labels = compute_short_execution_labels_for_panel(panel, keys, max_exit_scan_days=10)
            frame = pd.concat([keys.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
            path = output / f"trade_date={date}" / "part-000.parquet"
            path.parent.mkdir(parents=True)
            frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
            rows += len(frame)
            settled += int(frame["short_settled"].fillna(False).astype(bool).sum())
            hashes[date] = _file_sha256(path)
        return {
            "status": "pass" if settled == rows else "pass_with_unsettled",
            "rows": rows,
            "settled_rows": settled,
            "unsettled_rows": rows - settled,
            "unsettled_policy": "retain in the full-market ledger and exclude from realized-return evaluation until a legal exit settles",
            "partition_sha256": hashes,
        }

    @staticmethod
    def _load_blind(feature_dir: Path, label_dir: Path, dates: list[str]) -> pd.DataFrame:
        frames = []
        for date in dates:
            features = pd.read_parquet(feature_dir / f"trade_date={date}" / "part-000.parquet")
            labels = pd.read_parquet(label_dir / f"trade_date={date}" / "part-000.parquet")
            frames.append(features.merge(labels, on=["trade_date", "ts_code"], validate="one_to_one"))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _distribution_audit(train: pd.DataFrame, blind: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        dates = sorted(train["trade_date"].astype(str).unique())[-20:]
        reference = train[train["trade_date"].astype(str).isin(dates)]
        details = []
        for column in feature_columns:
            historical = pd.to_numeric(reference[column], errors="coerce")
            forward = pd.to_numeric(blind[column], errors="coerce")
            median = historical.median()
            q1, q3 = historical.quantile([0.25, 0.75])
            scale = float(q3 - q1)
            if not np.isfinite(scale) or scale <= 0:
                scale = float(historical.std())
            shift = None
            if pd.notna(median) and forward.notna().any():
                shift = abs(float(forward.median()) - float(median)) / (scale if np.isfinite(scale) and scale > 0 else 1.0)
            details.append(
                {
                    "feature": column,
                    "robust_median_shift": round(float(shift), 8) if shift is not None else None,
                    "reference_missing_rate": round(float(historical.isna().mean()), 8),
                    "blind_missing_rate": round(float(forward.isna().mean()), 8),
                }
            )
        severe_shift = [row for row in details if (row["robust_median_shift"] or 0.0) > 5.0]
        missing_shift = [
            row for row in details
            if abs(row["blind_missing_rate"] - row["reference_missing_rate"]) > 0.20
        ]
        ordered = sorted(details, key=lambda row: row["robust_median_shift"] or -1.0, reverse=True)
        return {
            "status": "pass" if not severe_shift and not missing_shift else "blocked",
            "reference_signal_dates": {"start": dates[0], "end": dates[-1], "count": len(dates)},
            "severe_shift_threshold": 5.0,
            "missing_rate_delta_threshold": 0.20,
            "severe_shift_features": [row["feature"] for row in severe_shift],
            "missing_shift_features": [row["feature"] for row in missing_shift],
            "top_shifts": ordered[:20],
            "policy": "data-contract diagnostic only; these thresholds do not select stocks or tune the model",
        }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
