from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from ..db import DailyQuote, session_scope
from ..factor_lab.metrics import max_drawdown
from .feature_policy import is_optional_feature


DEFAULT_TOP_N = (10, 20, 50)
DEFAULT_BASE_MODELS = ("lightgbm",)
DEFAULT_CROSS_TERMS = (
    ("feature_return_5d", "feature_volume_ratio"),
    ("feature_main_net_to_amount", "feature_relative_industry_strength_20d"),
    ("feature_return_20d", "feature_distance_to_20d_high"),
    ("feature_volume_ratio", "feature_turnover_rate"),
    ("feature_main_net_to_circ_mv", "feature_relative_industry_strength_20d"),
)
REPORT_CONTRACT_VERSION = "ml_research_contract_v2"
DEFAULT_IMPORTANCE_SAMPLE_ROWS = 1000
DEFAULT_IMPORTANCE_REPEATS = 3
DEFAULT_IMPORTANCE_TOP_N = 20
MARKET_STATE_FEATURES = (
    "feature_market_return_1d",
    "feature_market_return_5d",
    "feature_market_return_20d",
    "feature_market_drawdown_20d",
    "feature_market_volatility_20d",
    "feature_market_breadth_return_5d",
    "feature_market_breadth_flow_5d",
    "feature_market_return_dispersion_5d",
    "feature_limit_pressure_5d",
)


class MlResearchError(RuntimeError):
    """Raised when ML research experiments cannot be evaluated safely."""


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 120
    validation_days: int = 20
    step_days: int = 20
    top_n: tuple[int, ...] = DEFAULT_TOP_N
    base_models: tuple[str, ...] = DEFAULT_BASE_MODELS
    max_rows: int | None = None
    random_state: int = 42
    diagnostic_sample_rows: int = DEFAULT_IMPORTANCE_SAMPLE_ROWS
    diagnostic_repeats: int = DEFAULT_IMPORTANCE_REPEATS
    diagnostic_top_n: int = DEFAULT_IMPORTANCE_TOP_N


@dataclass(frozen=True)
class FoldWindow:
    fold_index: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


class MlResearchHarness:
    """Walk-forward research harness for ranking-oriented A-share ML experiments."""

    VERDICT_POLICY = "provisional_research_only; calibrate thresholds after full-distribution review before production use"

    def run_from_training_frame(
        self,
        training_frame_path: str | Path,
        output_dir: str | Path = "reports/ml_lab/research_harness",
        config: WalkForwardConfig | None = None,
        report_context: dict[str, Any] | None = None,
    ) -> dict:
        config = config or WalkForwardConfig()
        frame = self.load_training_frame(training_frame_path, max_rows=config.max_rows)
        enriched = self.add_cross_features(frame)
        feature_cols = self.feature_columns(enriched)
        folds = self.walk_forward_windows(enriched["trade_date"], config)
        if not folds:
            raise MlResearchError("No walk-forward folds can be created from the requested frame and window sizes.")

        evaluation_regimes = self.build_evaluation_regimes(enriched)
        results: list[dict[str, Any]] = []
        for model_name in config.base_models:
            for target_name in ("stable", "aggressive"):
                permutation_consumed = False
                for fold in folds:
                    result = self.evaluate_fold(
                        enriched,
                        feature_cols,
                        fold,
                        model_name,
                        target_name,
                        config,
                        evaluation_regimes.get("date_regimes", {}),
                        include_permutation_importance=not permutation_consumed,
                    )
                    results.append(result)
                    if result.get("status") == "trained" and not permutation_consumed:
                        permutation_consumed = True

        summary = self.summarize(results)
        regime_summary = self.summarize_regime_metrics(results)
        oos_daily_selection = [
            row
            for result in results
            for row in result.get("oos_daily_selection", [])
            if result.get("model") == "lightgbm" and result.get("target") == "stable"
        ]
        payload = {
            "task": "ml_research_harness",
            "report_contract_version": REPORT_CONTRACT_VERSION,
            "training_frame_path": str(training_frame_path),
            "config": asdict(config),
            "feature_count": len(feature_cols),
            "fold_count": len(folds),
            "folds": [asdict(item) for item in folds],
            "evaluation_regimes": evaluation_regimes,
            "results": results,
            "summary": summary,
            "regime_summary": regime_summary,
            "oos_daily_selection": oos_daily_selection,
            "diagnostic_summary": self.summarize_diagnostics(results),
            "failure_windows": self.identify_failure_windows(results),
            "verdict_policy": self.VERDICT_POLICY,
        }
        if report_context:
            payload.update(report_context)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "research_harness_report.json"
        summary_path = output_path / "research_harness_summary.csv"
        oos_path = output_path / "oos_daily_selection.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(oos_daily_selection).to_csv(oos_path, index=False, encoding="utf-8-sig")
        return {**payload, "report_path": str(report_path), "summary_path": str(summary_path), "oos_daily_selection_path": str(oos_path)}

    @staticmethod
    def load_training_frame(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
        frame = pd.read_csv(path, nrows=max_rows)
        if frame.empty:
            raise MlResearchError("Training frame is empty.")
        required = {"trade_date", "ts_code", "label_tradable", "label_net_return", "label_stable_positive", "label_top10"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise MlResearchError(f"Training frame missing required columns: {missing}")
        frame = frame.copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["label_net_return"] = pd.to_numeric(frame["label_net_return"], errors="coerce")
        tradable = frame["label_tradable"].astype(str).str.lower().isin({"true", "1"})
        return frame[tradable & frame["label_net_return"].notna()].reset_index(drop=True)

    @staticmethod
    def add_cross_features(frame: pd.DataFrame, cross_terms: tuple[tuple[str, str], ...] = DEFAULT_CROSS_TERMS) -> pd.DataFrame:
        result = frame.copy()
        for left, right in cross_terms:
            if left not in result.columns or right not in result.columns:
                continue
            left_values = pd.to_numeric(result[left], errors="coerce")
            right_values = pd.to_numeric(result[right], errors="coerce")
            result[f"cross__{left.replace('feature_', '')}__x__{right.replace('feature_', '')}"] = left_values * right_values
        return result

    @staticmethod
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        cols = [col for col in frame.columns if col.startswith("feature_") or col.startswith("cross__")]
        if not cols:
            raise MlResearchError("No feature columns found. Expected feature_* columns from ML Lab training frame.")
        return sorted(cols)

    @staticmethod
    def walk_forward_windows(trade_dates: pd.Series, config: WalkForwardConfig) -> list[FoldWindow]:
        dates = sorted(pd.Series(trade_dates.astype(str).unique()).dropna().tolist())
        windows: list[FoldWindow] = []
        total = len(dates)
        index = 0
        fold_index = 0
        while index + config.train_days + config.validation_days <= total:
            train = dates[index : index + config.train_days]
            valid = dates[index + config.train_days : index + config.train_days + config.validation_days]
            windows.append(
                FoldWindow(
                    fold_index=fold_index,
                    train_start=train[0],
                    train_end=train[-1],
                    validation_start=valid[0],
                    validation_end=valid[-1],
                )
            )
            fold_index += 1
            index += config.step_days
        return windows

    def evaluate_fold(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        fold: FoldWindow,
        model_name: str,
        target_name: str,
        config: WalkForwardConfig,
        date_regimes: dict[str, str] | None = None,
        *,
        include_permutation_importance: bool = False,
    ) -> dict:
        raw_train = frame[(frame["trade_date"] >= fold.train_start) & (frame["trade_date"] <= fold.train_end)].copy()
        train, split_audit = self.apply_label_embargo(raw_train, fold.validation_start)
        valid = frame[(frame["trade_date"] >= fold.validation_start) & (frame["trade_date"] <= fold.validation_end)].copy()
        if train.empty or valid.empty:
            return {"model": model_name, "target": target_name, "fold": asdict(fold), "status": "empty_fold", "split_audit": split_audit}

        x_train = self.clean_features(train, feature_cols)
        x_valid = self.clean_features(valid, feature_cols)
        y_train = self.target_values(train, target_name)
        usable = y_train.notna()
        x_train = x_train.loc[usable]
        y_train = y_train.loc[usable]
        if y_train.nunique(dropna=True) < 2 and target_name == "stable":
            return {"model": model_name, "target": target_name, "fold": asdict(fold), "status": "single_class_target", "split_audit": split_audit}

        model = self.make_model(model_name, target_name, random_state=config.random_state)
        model.fit(x_train, y_train)
        train_eval = train.loc[x_train.index].copy()
        train_eval["prediction"] = self.predict_score(model, x_train, target_name)
        valid = valid.copy()
        valid["prediction"] = self.predict_score(model, x_valid, target_name)
        metrics = self.evaluate_predictions(valid, config.top_n)
        train_metrics = self.evaluate_predictions(train_eval, config.top_n)
        diagnostics = self.fold_diagnostics(
            model,
            valid,
            feature_cols,
            target_name,
            config,
            include_permutation_importance=include_permutation_importance,
        )
        oos_daily_selection = self.daily_oos_selection(valid, fold, target_name)
        return {
            "model": model_name,
            "target": target_name,
            "fold": asdict(fold),
            "status": "trained",
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(valid)),
            "split_audit": split_audit,
            "train_metrics": train_metrics,
            "metrics": metrics,
            "regime_metrics": self.evaluate_predictions_by_regime(valid, config.top_n, date_regimes or {}),
            "oos_daily_selection": oos_daily_selection,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def daily_oos_selection(frame: pd.DataFrame, fold: FoldWindow, target_name: str) -> list[dict]:
        """Persist only the OOS stable Top10 stream used by the v6 meta-gate."""
        if target_name != "stable" or frame.empty:
            return []
        daily = MlResearchHarness._daily_topn_prediction_rows(frame, (10,))
        if daily.empty:
            return []
        state_columns = [column for column in MARKET_STATE_FEATURES if column in frame.columns]
        state = frame.groupby("trade_date", as_index=False)[state_columns].median(numeric_only=True) if state_columns else pd.DataFrame({"trade_date": daily["trade_date"]})
        merged = daily.merge(state, on="trade_date", how="left")
        merged["fold_index"] = fold.fold_index
        merged["validation_start"] = fold.validation_start
        merged["validation_end"] = fold.validation_end
        merged["selection_positive"] = (merged["mean_net_return"] > 0).astype(int)
        return merged.to_dict("records")

    def run_audit_from_training_frame(
        self,
        training_frame_path: str | Path,
        output_dir: str | Path = "reports/ml_lab/research_audit",
        config: WalkForwardConfig | None = None,
        model_report_path: str | Path | None = None,
        report_context: dict[str, Any] | None = None,
    ) -> dict:
        config = config or WalkForwardConfig(base_models=())
        raw = pd.read_csv(training_frame_path, nrows=config.max_rows)
        label_audit = self.audit_training_frame(raw)
        frame = self.load_training_frame(training_frame_path, max_rows=config.max_rows)
        folds = self.walk_forward_windows(frame["trade_date"], config)
        if not folds:
            raise MlResearchError("No walk-forward folds can be created for audit.")
        split_audit = self.audit_walk_forward_splits(frame, folds)

        evaluation_regimes = self.build_evaluation_regimes(frame)
        baseline_results = self.evaluate_baselines(frame, folds, config, evaluation_regimes.get("date_regimes", {}))
        baseline_summary = self.summarize(baseline_results)
        baseline_regime_summary = self.summarize_regime_metrics(baseline_results)
        model_summary = self._load_model_summary(model_report_path)
        baseline_comparison = self.build_baseline_comparison(baseline_summary, model_summary)
        gate = self.research_gate(label_audit, baseline_summary, model_summary, split_audit=split_audit)
        payload = {
            "task": "ml_research_audit",
            "report_contract_version": REPORT_CONTRACT_VERSION,
            "training_frame_path": str(training_frame_path),
            "model_report_path": str(model_report_path) if model_report_path else None,
            "config": asdict(config),
            "label_audit": label_audit,
            "split_audit": split_audit,
            "evaluation_regimes": evaluation_regimes,
            "baseline_results": baseline_results,
            "baseline_summary": baseline_summary,
            "baseline_regime_summary": baseline_regime_summary,
            "baseline_failure_windows": self.identify_failure_windows(baseline_results),
            "baseline_comparison": baseline_comparison,
            "model_summary": model_summary,
            "gate": gate,
        }
        if report_context:
            payload.update(report_context)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "research_audit_report.json"
        baseline_path = output_path / "baseline_summary.csv"
        baseline_regime_path = output_path / "baseline_regime_summary.csv"
        baseline_failure_path = output_path / "baseline_failure_windows.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(baseline_summary).to_csv(baseline_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(baseline_regime_summary).to_csv(baseline_regime_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(payload["baseline_failure_windows"]).to_csv(baseline_failure_path, index=False, encoding="utf-8-sig")
        return {
            **payload,
            "report_path": str(report_path),
            "baseline_summary_path": str(baseline_path),
            "baseline_regime_summary_path": str(baseline_regime_path),
            "baseline_failure_windows_path": str(baseline_failure_path),
        }

    def run_label_reconstruction_audit(
        self,
        training_frame_path: str | Path,
        output_dir: str | Path = "reports/ml_lab/label_reconstruction_audit",
        sample_size: int = 1000,
        random_state: int = 42,
        tolerance: float = 1e-9,
    ) -> dict:
        frame = pd.read_csv(
            training_frame_path,
            usecols=[
                "trade_date",
                "ts_code",
                "label_buy_date",
                "label_sell_date",
                "label_tradable",
                "label_net_return",
            ],
        )
        sample = self.sample_label_rows(frame, sample_size=sample_size, random_state=random_state)
        audit = self.reconstruct_labels_from_db(sample, tolerance=tolerance)
        payload = {
            "task": "label_reconstruction_audit",
            "training_frame_path": str(training_frame_path),
            "sample_size_requested": sample_size,
            "sample_size": int(len(sample)),
            "random_state": random_state,
            "tolerance": tolerance,
            **audit,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "label_reconstruction_audit.json"
        mismatch_path = output_path / "label_reconstruction_mismatches.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(audit["mismatches"]).to_csv(mismatch_path, index=False, encoding="utf-8-sig")
        return {**payload, "report_path": str(report_path), "mismatch_path": str(mismatch_path)}

    @staticmethod
    def sample_label_rows(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
        required = {"trade_date", "ts_code", "label_buy_date", "label_sell_date", "label_tradable", "label_net_return"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise MlResearchError(f"Training frame missing reconstruction columns: {missing}")
        clean = frame.dropna(subset=["trade_date", "ts_code", "label_buy_date", "label_sell_date"]).copy()
        for column in ("trade_date", "label_buy_date", "label_sell_date"):
            clean[column] = clean[column].map(MlResearchHarness._normalize_yyyymmdd)
        clean["ts_code"] = clean["ts_code"].astype(str).str.strip()
        clean = clean[
            (clean["trade_date"] != "")
            & (clean["ts_code"] != "")
            & (clean["label_buy_date"] != "")
            & (clean["label_sell_date"] != "")
        ]
        if clean.empty:
            raise MlResearchError("No label rows available for reconstruction audit.")
        size = min(sample_size, len(clean))
        return clean.sample(n=size, random_state=random_state).reset_index(drop=True)

    @staticmethod
    def reconstruct_labels_from_db(sample: pd.DataFrame, tolerance: float = 1e-9) -> dict:
        quote_map = MlResearchHarness._load_audit_quotes(sample)
        cost = 0.0003 + 0.0003 + 0.001 + 0.0005 * 2
        mismatches: list[dict[str, Any]] = []
        checked = 0
        missing_quote_rows = 0
        tradable_mismatches = 0
        return_mismatches = 0
        nontradable_reason_counts: dict[str, int] = {}

        for row in sample.to_dict("records"):
            ts_code = str(row["ts_code"]).strip()
            trade_date = MlResearchHarness._normalize_yyyymmdd(row["trade_date"])
            buy_date = MlResearchHarness._normalize_yyyymmdd(row["label_buy_date"])
            sell_date = MlResearchHarness._normalize_yyyymmdd(row["label_sell_date"])
            expected_tradable = MlResearchHarness._coerce_label_bool(row.get("label_tradable"))
            expected_return = pd.to_numeric(pd.Series([row.get("label_net_return")]), errors="coerce").iloc[0]
            buy_quote = quote_map.get((ts_code, buy_date))
            sell_quote = quote_map.get((ts_code, sell_date))
            reconstructed = MlResearchHarness._reconstruct_one_label(buy_quote, sell_quote, cost)
            checked += 1

            if reconstructed["missing_quote"]:
                missing_quote_rows += 1
            if not reconstructed["tradable"]:
                for reason in reconstructed["reasons"]:
                    nontradable_reason_counts[reason] = nontradable_reason_counts.get(reason, 0) + 1

            return_match = True
            diff = None
            if reconstructed["tradable"]:
                if pd.isna(expected_return):
                    return_match = False
                    diff = None
                else:
                    diff = float(reconstructed["label_net_return"] - float(expected_return))
                    return_match = abs(diff) <= tolerance
            elif not pd.isna(expected_return):
                return_match = False

            tradable_match = expected_tradable == reconstructed["tradable"]
            if not tradable_match:
                tradable_mismatches += 1
            if not return_match:
                return_mismatches += 1
            if not tradable_match or not return_match:
                mismatches.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "label_buy_date": buy_date,
                        "label_sell_date": sell_date,
                        "expected_tradable": expected_tradable,
                        "reconstructed_tradable": reconstructed["tradable"],
                        "expected_return": None if pd.isna(expected_return) else float(expected_return),
                        "reconstructed_return": reconstructed["label_net_return"],
                        "return_diff": diff,
                        "reasons": ";".join(reconstructed["reasons"]),
                    }
                )

        status = "pass" if tradable_mismatches == 0 and return_mismatches == 0 else "review"
        return {
            "checked_rows": checked,
            "status": status,
            "missing_quote_rows": missing_quote_rows,
            "tradable_mismatches": tradable_mismatches,
            "return_mismatches": return_mismatches,
            "mismatch_rate": round(len(mismatches) / checked, 6) if checked else None,
            "nontradable_reason_counts": nontradable_reason_counts,
            "missing_quote_policy": "Missing buy/sell quotes are valid non-tradable reasons when the exported label also marks the row non-tradable; only tradability or return mismatches block the audit.",
            "mismatches": mismatches[:200],
        }

    @staticmethod
    def _load_audit_quotes(sample: pd.DataFrame) -> dict[tuple[str, str], dict]:
        needed = {
            (str(row["ts_code"]).strip(), MlResearchHarness._normalize_yyyymmdd(row["label_buy_date"]))
            for row in sample.to_dict("records")
        } | {
            (str(row["ts_code"]).strip(), MlResearchHarness._normalize_yyyymmdd(row["label_sell_date"]))
            for row in sample.to_dict("records")
        }
        by_code: dict[str, set[str]] = {}
        for ts_code, trade_date in needed:
            if not ts_code or not trade_date:
                continue
            by_code.setdefault(ts_code, set()).add(trade_date)

        quote_map: dict[tuple[str, str], dict] = {}
        with session_scope() as session:
            for ts_code, dates in by_code.items():
                rows = (
                    session.query(DailyQuote)
                    .filter(DailyQuote.ts_code == ts_code, DailyQuote.trade_date.in_(sorted(dates)))
                    .all()
                )
                for quote in rows:
                    quote_map[(quote.ts_code, quote.trade_date)] = {
                        "open": quote.open,
                        "close": quote.close,
                        "pct_chg": quote.pct_chg,
                    }
        return quote_map

    @staticmethod
    def _normalize_yyyymmdd(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return str(int(value)) if float(value).is_integer() else str(value).strip()
        text = str(value).strip()
        if not text:
            return ""
        numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if not pd.isna(numeric) and float(numeric).is_integer():
            return str(int(numeric))
        if text.endswith(".0"):
            return text[:-2]
        return text

    @staticmethod
    def _coerce_label_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) != 0.0
        return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}

    @staticmethod
    def _reconstruct_one_label(buy_quote: dict | None, sell_quote: dict | None, cost: float) -> dict:
        reasons = []
        if buy_quote is None:
            reasons.append("missing_buy_quote")
        if sell_quote is None:
            reasons.append("missing_sell_quote")
        if buy_quote and buy_quote.get("open") in (None, 0):
            reasons.append("missing_buy_open")
        if sell_quote and sell_quote.get("close") in (None, 0):
            reasons.append("missing_sell_close")
        if buy_quote and buy_quote.get("pct_chg") is not None and float(buy_quote["pct_chg"]) >= 9.8:
            reasons.append("limit_up_entry_block")
        if sell_quote and sell_quote.get("pct_chg") is not None and float(sell_quote["pct_chg"]) <= -9.8:
            reasons.append("limit_down_exit_block")

        tradable = not reasons
        net_return = None
        if tradable:
            net_return = float(sell_quote["close"]) / float(buy_quote["open"]) - 1.0 - cost
        return {
            "tradable": tradable,
            "label_net_return": net_return,
            "missing_quote": buy_quote is None or sell_quote is None,
            "reasons": reasons,
        }

    @staticmethod
    def audit_training_frame(frame: pd.DataFrame) -> dict:
        required = {"trade_date", "ts_code", "label_tradable", "label_net_return", "label_stable_positive", "label_top10"}
        missing_required = sorted(required - set(frame.columns))
        feature_cols = sorted([col for col in frame.columns if col.startswith("feature_") or col.startswith("cross__")])
        trade_dates = frame["trade_date"].astype(str) if "trade_date" in frame.columns else pd.Series(dtype=str)
        tradable = frame["label_tradable"].astype(str).str.lower().isin({"true", "1"}) if "label_tradable" in frame.columns else pd.Series([False] * len(frame))
        net = pd.to_numeric(frame.get("label_net_return"), errors="coerce") if "label_net_return" in frame.columns else pd.Series(dtype=float)

        date_order = {"checked": False, "violations": 0, "buy_not_after_signal": 0, "sell_before_buy": 0}
        if {"trade_date", "label_buy_date", "label_sell_date"}.issubset(frame.columns):
            signal = frame["trade_date"].astype(str)
            buy = frame["label_buy_date"].astype(str)
            sell = frame["label_sell_date"].astype(str)
            buy_not_after_signal = (buy <= signal).sum()
            sell_before_buy = (sell < buy).sum()
            date_order = {
                "checked": True,
                "violations": int(buy_not_after_signal + sell_before_buy),
                "buy_not_after_signal": int(buy_not_after_signal),
                "sell_before_buy": int(sell_before_buy),
            }

        suspicious_terms = ("future", "forward", "label", "buy_date", "sell_date", "net_return")
        suspicious_feature_columns = [
            col
            for col in feature_cols
            if any(term in col.lower() for term in suspicious_terms)
        ]
        sensitive_column_warnings = [
            col
            for col in frame.columns
            if any(term in col.lower() for term in ("token", "password", "secret", "account", "cash"))
        ]
        missingness = []
        for col in feature_cols:
            missingness.append({"column": col, "missing_rate": round(float(frame[col].isna().mean()), 6)})
        missingness.sort(key=lambda row: row["missing_rate"], reverse=True)
        feature_missingness_gate = MlResearchHarness.feature_missingness_gate(frame, feature_cols)

        stable = pd.to_numeric(frame.get("label_stable_positive"), errors="coerce") if "label_stable_positive" in frame.columns else pd.Series(dtype=float)
        top10 = pd.to_numeric(frame.get("label_top10"), errors="coerce") if "label_top10" in frame.columns else pd.Series(dtype=float)
        issues = []
        if missing_required:
            issues.append("missing_required_columns")
        if date_order["checked"] and date_order["violations"] > 0:
            issues.append("label_date_order_violations")
        if suspicious_feature_columns:
            issues.append("suspicious_feature_column_names")
        if sensitive_column_warnings:
            issues.append("sensitive_column_names")
        if not feature_cols:
            issues.append("no_feature_columns")
        if feature_missingness_gate["status"] != "pass":
            issues.append("feature_missingness_gate_blocked")

        return {
            "rows": int(len(frame)),
            "unique_dates": int(trade_dates.nunique()) if not trade_dates.empty else 0,
            "date_range": {
                "start": str(trade_dates.min()) if not trade_dates.empty else None,
                "end": str(trade_dates.max()) if not trade_dates.empty else None,
            },
            "unique_codes": int(frame["ts_code"].astype(str).nunique()) if "ts_code" in frame.columns else 0,
            "feature_count": len(feature_cols),
            "missing_required_columns": missing_required,
            "tradable_rows": int(tradable.sum()) if len(tradable) else 0,
            "tradable_rate": round(float(tradable.mean()), 6) if len(tradable) else None,
            "label_net_return_non_null_rate": round(float(net.notna().mean()), 6) if len(net) else None,
            "label_net_return_mean_tradable": round(float(net[tradable].mean()), 6) if len(net) and tradable.any() else None,
            "stable_positive_rate": round(float(stable.mean()), 6) if len(stable) else None,
            "top10_rate": round(float(top10.mean()), 6) if len(top10) else None,
            "label_date_order": date_order,
            "suspicious_feature_columns": suspicious_feature_columns,
            "sensitive_column_warnings": sensitive_column_warnings,
            "feature_missingness_top10": missingness[:10],
            "feature_missingness_gate": feature_missingness_gate,
            "issues": issues,
            "status": "pass" if not issues else "review",
        }

    @staticmethod
    def feature_missingness_gate(
        frame: pd.DataFrame,
        feature_cols: list[str] | None = None,
        required_max_missing_rate: float = 0.05,
    ) -> dict:
        cols = feature_cols or sorted([col for col in frame.columns if col.startswith("feature_") or col.startswith("cross__")])
        rows = []
        blocking = []
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(dtype=float)
            missing_rate = float(values.isna().mean()) if len(values) else 1.0
            optional = MlResearchHarness.is_optional_feature(col)
            threshold = 0.20 if optional else required_max_missing_rate
            item = {
                "column": col,
                "missing_rate": round(missing_rate, 6),
                "non_null_rows": int(values.notna().sum()),
                "required": not optional,
                "threshold": threshold,
                "status": "pass" if missing_rate <= threshold else "blocked",
            }
            rows.append(item)
            if item["status"] != "pass":
                blocking.append(item)
        rows.sort(key=lambda row: (row["status"] == "pass", -row["missing_rate"], row["column"]))
        return {
            "status": "pass" if not blocking else "blocked",
            "feature_count": len(cols),
            "required_max_missing_rate": required_max_missing_rate,
            "optional_max_missing_rate": 0.20,
            "blocking_count": len(blocking),
            "blocking_features": blocking,
            "feature_missingness": rows,
            "policy": "Required active training features must not exceed the missingness threshold; 100%-missing features are blocked.",
        }

    @staticmethod
    def is_optional_feature(name_or_column: str) -> bool:
        return is_optional_feature(name_or_column)

    @staticmethod
    def apply_label_embargo(frame: pd.DataFrame, validation_start: str) -> tuple[pd.DataFrame, dict]:
        validation_start = MlResearchHarness._normalize_yyyymmdd(validation_start)
        label_columns = [column for column in ("label_buy_date", "label_sell_date") if column in frame.columns]
        if not label_columns:
            return frame.copy(), {
                "checked": False,
                "validation_start": validation_start,
                "input_rows": int(len(frame)),
                "output_rows": int(len(frame)),
                "embargoed_rows": 0,
                "post_embargo_label_overlap_rows": None,
                "issue": "missing_label_date_columns",
                "policy": "Training rows should be embargoed when their buy/sell label dates touch the validation window.",
            }

        overlap = pd.Series(False, index=frame.index)
        for column in label_columns:
            dates = frame[column].map(MlResearchHarness._normalize_yyyymmdd)
            overlap = overlap | ((dates != "") & (dates >= validation_start))
        clean = frame.loc[~overlap].copy()

        post_overlap = pd.Series(False, index=clean.index)
        for column in label_columns:
            dates = clean[column].map(MlResearchHarness._normalize_yyyymmdd)
            post_overlap = post_overlap | ((dates != "") & (dates >= validation_start))

        return clean, {
            "checked": True,
            "validation_start": validation_start,
            "input_rows": int(len(frame)),
            "output_rows": int(len(clean)),
            "embargoed_rows": int(overlap.sum()),
            "post_embargo_label_overlap_rows": int(post_overlap.sum()) if len(clean) else 0,
            "policy": "Training rows whose buy/sell label dates reach the validation window are excluded before fitting.",
        }

    @staticmethod
    def audit_walk_forward_splits(frame: pd.DataFrame, folds: list[FoldWindow]) -> dict:
        issues: list[str] = []
        fold_rows = []
        for fold in folds:
            raw_train = frame[(frame["trade_date"] >= fold.train_start) & (frame["trade_date"] <= fold.train_end)].copy()
            valid = frame[(frame["trade_date"] >= fold.validation_start) & (frame["trade_date"] <= fold.validation_end)].copy()
            _, embargo = MlResearchHarness.apply_label_embargo(raw_train, fold.validation_start)
            row = {
                "fold_index": fold.fold_index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "validation_rows": int(len(valid)),
                **embargo,
            }
            if not embargo.get("checked"):
                issues.append(f"fold_{fold.fold_index}_missing_label_date_columns")
            if embargo.get("output_rows", 0) <= 0:
                issues.append(f"fold_{fold.fold_index}_empty_train_after_label_embargo")
            if embargo.get("post_embargo_label_overlap_rows") not in (0, None):
                issues.append(f"fold_{fold.fold_index}_label_overlap_after_embargo")
            if len(valid) <= 0:
                issues.append(f"fold_{fold.fold_index}_empty_validation")
            fold_rows.append(row)
        return {
            "status": "pass" if not issues else "review",
            "fold_count": len(folds),
            "issues": issues,
            "folds": fold_rows,
            "policy": "Walk-forward training labels must be fully known before each validation window starts.",
        }

    def evaluate_baselines(self, frame: pd.DataFrame, folds: list[FoldWindow], config: WalkForwardConfig, date_regimes: dict[str, str] | None = None) -> list[dict]:
        baseline_specs = self.baseline_specs(frame)
        results: list[dict[str, Any]] = []
        rng = np.random.default_rng(config.random_state)
        random_values = pd.Series(rng.random(len(frame)), index=frame.index)
        date_regimes = date_regimes or {}
        for fold in folds:
            valid = frame[(frame["trade_date"] >= fold.validation_start) & (frame["trade_date"] <= fold.validation_end)].copy()
            if valid.empty:
                continue
            for name, column, ascending in baseline_specs:
                scored = valid.copy()
                if name == "random":
                    scored["prediction"] = random_values.loc[scored.index]
                else:
                    scored["prediction"] = pd.to_numeric(scored[column], errors="coerce")
                    if ascending:
                        scored["prediction"] = -scored["prediction"]
                scored["prediction"] = scored["prediction"].replace([np.inf, -np.inf], np.nan).fillna(scored["prediction"].median())
                if scored["prediction"].isna().all():
                    continue
                scored["prediction"] = scored["prediction"].fillna(0.0)
                results.append(
                    {
                        "model": f"baseline:{name}",
                        "target": "net_return",
                        "fold": asdict(fold),
                        "status": "trained",
                        "train_rows": 0,
                        "validation_rows": int(len(scored)),
                        "metrics": self.evaluate_predictions(scored, config.top_n),
                        "regime_metrics": self.evaluate_predictions_by_regime(scored, config.top_n, date_regimes),
                    }
                )
        return results

    @staticmethod
    def baseline_specs(frame: pd.DataFrame) -> list[tuple[str, str | None, bool]]:
        candidates: list[tuple[str, str | None, bool]] = [("random", None, False)]
        for name, column, ascending in [
            ("rule_stable_score", "stable_score", False),
            ("rule_aggressive_score", "aggressive_score", False),
            ("return_5d", "feature_return_5d", False),
            ("return_20d", "feature_return_20d", False),
            ("volume_ratio", "feature_volume_ratio", False),
            ("relative_industry_strength_20d", "feature_relative_industry_strength_20d", False),
            ("main_net_to_amount", "feature_main_net_to_amount", False),
            ("low_volatility_20d", "feature_volatility_20d", True),
        ]:
            if column in frame.columns:
                candidates.append((name, column, ascending))
        return candidates

    @staticmethod
    def _load_model_summary(model_report_path: str | Path | None) -> list[dict]:
        if not model_report_path:
            return []
        path = Path(model_report_path)
        if not path.exists():
            return []
        report = json.loads(path.read_text(encoding="utf-8"))
        return report.get("summary") or report.get("validation_summary", {}).get("windows") or []

    @staticmethod
    def research_gate(label_audit: dict, baseline_summary: list[dict], model_summary: list[dict], split_audit: dict | None = None) -> dict:
        reasons = []
        status = "observe_only"
        if label_audit.get("status") != "pass":
            status = "rejected"
            reasons.extend(label_audit.get("issues") or [])
        if split_audit is not None and split_audit.get("status") != "pass":
            status = "rejected"
            reasons.extend(split_audit.get("issues") or ["split_audit_not_pass"])
        if not baseline_summary:
            status = "rejected"
            reasons.append("missing_baseline_summary")
        if not model_summary:
            reasons.append("missing_model_summary_for_incremental_comparison")
        best_baseline = MlResearchHarness._best_summary_row(baseline_summary)
        best_model = MlResearchHarness._best_summary_row(model_summary)
        comparison = MlResearchHarness.build_baseline_comparison(baseline_summary, model_summary)
        if status != "rejected" and best_model:
            model_return = float(best_model.get("avg_mean_net_return") or 0.0)
            positive_rate = float(best_model.get("positive_fold_rate") or 0.0)
            beats_all = bool(comparison.get("beats_all_baseline_categories"))
            risk_tags = best_model.get("risk_tags") or []
            if not beats_all:
                status = "rejected"
                reasons.append("model_does_not_beat_all_baseline_categories")
            elif model_return <= 0:
                status = "observe_only"
                reasons.append("best_model_avg_return_not_positive")
            elif positive_rate >= 0.7 and not risk_tags:
                status = "usable"
            elif positive_rate >= 0.6:
                status = "candidate"
            else:
                status = "observe_only"
                reasons.append("positive_fold_rate_below_candidate_threshold")
        return {
            "status": status,
            "reasons": reasons,
            "best_baseline": best_baseline,
            "best_model": best_model,
            "baseline_comparison": comparison,
            "allowed_statuses": ["rejected", "observe_only", "candidate", "usable"],
            "policy": "models must beat random, rule, and simple-factor baselines before production ranking",
        }

    @staticmethod
    def build_baseline_comparison(baseline_summary: list[dict], model_summary: list[dict]) -> dict:
        best_baseline = MlResearchHarness._best_summary_row(baseline_summary)
        best_model = MlResearchHarness._best_summary_row(model_summary)
        if not best_baseline and not best_model:
            return {
                "status": "missing",
                "policy": "compare promoted research summaries against random, rule, and simple-factor baselines before production ranking",
                "best_baseline": None,
                "best_model": None,
                "model_minus_baseline": None,
                "top_n_agreement": None,
                "best_baseline_by_category": {},
                "model_minus_baseline_by_category": {},
                "beats_all_baseline_categories": False,
            }

        model_minus_baseline = None
        top_n_agreement = None
        if best_baseline and best_model:
            model_minus_baseline = round(float((best_model.get("avg_mean_net_return") or 0.0) - (best_baseline.get("avg_mean_net_return") or 0.0)), 6)
            top_n_agreement = bool(best_baseline.get("top_n") == best_model.get("top_n"))
        status = "model_ahead" if (model_minus_baseline or 0) > 0 else "baseline_ahead" if (model_minus_baseline or 0) < 0 else "tied"
        best_by_category = MlResearchHarness._best_baseline_by_category(baseline_summary)
        model_minus_by_category = {}
        if best_model:
            model_return = float(best_model.get("avg_mean_net_return") or 0.0)
            for category, row in best_by_category.items():
                model_minus_by_category[category] = round(model_return - float(row.get("avg_mean_net_return") or 0.0), 6)
        beats_all = bool(model_minus_by_category) and all(value > 0 for value in model_minus_by_category.values())
        return {
            "status": status,
            "policy": "compare promoted research summaries against random, rule, and simple-factor baselines before production ranking",
            "best_baseline": best_baseline,
            "best_model": best_model,
            "model_minus_baseline": model_minus_baseline,
            "top_n_agreement": top_n_agreement,
            "best_baseline_by_category": best_by_category,
            "model_minus_baseline_by_category": model_minus_by_category,
            "beats_all_baseline_categories": beats_all,
        }

    @staticmethod
    def _best_baseline_by_category(baseline_summary: list[dict]) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = {}
        for row in baseline_summary or []:
            model = str(row.get("model", ""))
            if not model.startswith("baseline:"):
                continue
            name = model.split(":", 1)[1]
            if name == "random":
                category = "random"
            elif name.startswith("rule_"):
                category = "rule"
            else:
                category = "simple_factor"
            grouped.setdefault(category, []).append(row)
        return {category: MlResearchHarness._best_summary_row(rows) for category, rows in grouped.items()}

    @staticmethod
    def clean_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def target_values(frame: pd.DataFrame, target_name: str) -> pd.Series:
        if target_name == "stable":
            return pd.to_numeric(frame["label_stable_positive"], errors="coerce").astype("Int64")
        if target_name == "aggressive":
            return pd.to_numeric(frame["label_net_return"], errors="coerce")
        raise ValueError(target_name)

    @staticmethod
    def make_model(model_name: str, target_name: str, random_state: int):
        if model_name == "lightgbm":
            if target_name == "stable":
                from lightgbm import LGBMClassifier

                return LGBMClassifier(n_estimators=180, learning_rate=0.04, max_depth=5, num_leaves=31, random_state=random_state, verbose=-1)
            from lightgbm import LGBMRegressor

            return LGBMRegressor(n_estimators=220, learning_rate=0.035, max_depth=5, num_leaves=31, random_state=random_state, verbose=-1)
        if model_name == "xgboost":
            if target_name == "stable":
                from xgboost import XGBClassifier

                return XGBClassifier(n_estimators=180, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.85, random_state=random_state, eval_metric="logloss")
            from xgboost import XGBRegressor

            return XGBRegressor(n_estimators=220, learning_rate=0.035, max_depth=4, subsample=0.85, colsample_bytree=0.85, random_state=random_state)
        if model_name == "catboost":
            if target_name == "stable":
                from catboost import CatBoostClassifier

                return CatBoostClassifier(iterations=180, learning_rate=0.04, depth=5, random_seed=random_state, verbose=False)
            from catboost import CatBoostRegressor

            return CatBoostRegressor(iterations=220, learning_rate=0.035, depth=5, random_seed=random_state, verbose=False)
        raise ValueError(model_name)

    @staticmethod
    def predict_score(model: Any, features: pd.DataFrame, target_name: str) -> np.ndarray:
        if target_name == "stable" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            return np.asarray([row[-1] for row in proba], dtype=float)
        return np.asarray(model.predict(features), dtype=float)

    @staticmethod
    def evaluate_predictions(frame: pd.DataFrame, top_n_values: tuple[int, ...]) -> dict:
        daily = MlResearchHarness._daily_topn_prediction_rows(frame, top_n_values)
        result: dict[str, dict] = {}
        for top_n in top_n_values:
            rows = daily[daily["top_n"] == top_n].sort_values("trade_date") if not daily.empty else pd.DataFrame()
            result[f"top{top_n}"] = MlResearchHarness._aggregate_daily_prediction_rows(rows)
        return result

    @staticmethod
    def evaluate_predictions_by_regime(frame: pd.DataFrame, top_n_values: tuple[int, ...], date_regimes: dict[str, str]) -> dict:
        daily = MlResearchHarness._daily_topn_prediction_rows(frame, top_n_values)
        if daily.empty:
            return {}
        daily = daily.copy()
        daily["evaluation_regime"] = daily["trade_date"].astype(str).map(date_regimes).fillna("unknown")
        result: dict[str, dict] = {}
        for top_n in top_n_values:
            top_rows = daily[daily["top_n"] == top_n]
            by_regime = {}
            for regime, rows in top_rows.groupby("evaluation_regime"):
                by_regime[str(regime)] = MlResearchHarness._aggregate_daily_prediction_rows(rows.sort_values("trade_date"))
            result[f"top{top_n}"] = by_regime
        return result

    @staticmethod
    def _daily_topn_prediction_rows(frame: pd.DataFrame, top_n_values: tuple[int, ...]) -> pd.DataFrame:
        ranked = frame.sort_values("prediction", ascending=False)
        by_date = []
        for trade_date, group in ranked.groupby("trade_date"):
            for top_n in top_n_values:
                top = group.head(top_n)
                if top.empty:
                    continue
                by_date.append(
                    {
                        "trade_date": trade_date,
                        "top_n": top_n,
                        "mean_net_return": float(top["label_net_return"].mean()),
                        "win_rate": float((top["label_net_return"] > 0).mean()),
                        "count": int(len(top)),
                    }
                )
        return pd.DataFrame(by_date)

    @staticmethod
    def _aggregate_daily_prediction_rows(rows: pd.DataFrame) -> dict:
        returns = rows["mean_net_return"].tolist() if not rows.empty else []
        mean_return = float(np.mean(returns)) if returns else None
        drawdown = max_drawdown(returns) if returns else None
        return {
            "days": int(len(rows)),
            "mean_net_return": round(mean_return, 6) if mean_return is not None else None,
            "win_rate": round(float(rows["win_rate"].mean()), 6) if not rows.empty else None,
            "positive_day_rate": round(float((rows["mean_net_return"] > 0).mean()), 6) if not rows.empty else None,
            "max_drawdown": round(float(drawdown), 6) if drawdown is not None else None,
            "return_drawdown_ratio": round(mean_return / abs(drawdown), 6) if mean_return is not None and drawdown not in (None, 0) else None,
        }

    @staticmethod
    def build_evaluation_regimes(frame: pd.DataFrame) -> dict:
        if not {"trade_date", "label_net_return"}.issubset(frame.columns):
            return {"policy": "missing trade_date or label_net_return; no evaluation regimes built", "date_regimes": {}, "daily": [], "summary": []}
        working = frame[["trade_date", "label_net_return"]].copy()
        working["trade_date"] = working["trade_date"].map(MlResearchHarness._normalize_yyyymmdd)
        working["label_net_return"] = pd.to_numeric(working["label_net_return"], errors="coerce")
        working = working[(working["trade_date"] != "") & working["label_net_return"].notna()]
        if working.empty:
            return {"policy": "empty label returns; no evaluation regimes built", "date_regimes": {}, "daily": [], "summary": []}

        daily = (
            working.groupby("trade_date")["label_net_return"]
            .agg(rows="count", mean_return="mean", median_return="median", dispersion="std", positive_rate=lambda s: float((s > 0).mean()))
            .reset_index()
            .sort_values("trade_date")
        )
        daily["dispersion"] = daily["dispersion"].fillna(0.0)
        high_dispersion_threshold = float(daily["dispersion"].quantile(0.75)) if len(daily) else 0.0

        def direction(row: pd.Series) -> str:
            mean_return = float(row["mean_return"])
            positive_rate = float(row["positive_rate"])
            if mean_return >= 0.005 and positive_rate >= 0.55:
                return "broad_up"
            if mean_return <= -0.005 and positive_rate <= 0.45:
                return "broad_down"
            return "mixed"

        daily["direction_regime"] = daily.apply(direction, axis=1)
        daily["dispersion_regime"] = np.where(daily["dispersion"] >= high_dispersion_threshold, "high_dispersion", "normal_dispersion")
        daily["evaluation_regime"] = daily["direction_regime"]
        date_regimes = dict(zip(daily["trade_date"].astype(str), daily["evaluation_regime"].astype(str), strict=False))
        summary = []
        for regime, rows in daily.groupby("evaluation_regime"):
            summary.append(
                {
                    "evaluation_regime": str(regime),
                    "days": int(len(rows)),
                    "avg_market_mean_return": round(float(rows["mean_return"].mean()), 6),
                    "avg_positive_rate": round(float(rows["positive_rate"].mean()), 6),
                    "avg_dispersion": round(float(rows["dispersion"].mean()), 6),
                }
            )
        return {
            "policy": "evaluation-only outcome regimes derived from cross-sectional future labels; do not use as live signal",
            "high_dispersion_threshold": round(high_dispersion_threshold, 6),
            "date_regimes": date_regimes,
            "daily": daily.round(
                {
                    "mean_return": 6,
                    "median_return": 6,
                    "dispersion": 6,
                    "positive_rate": 6,
                }
            ).to_dict("records"),
            "summary": sorted(summary, key=lambda row: row["evaluation_regime"]),
        }

    @staticmethod
    def summarize(results: list[dict]) -> list[dict]:
        rows = []
        for item in results:
            if item.get("status") != "trained":
                continue
            for top_key, metrics in (item.get("metrics") or {}).items():
                rows.append(
                    {
                        "model": item["model"],
                        "target": item["target"],
                        "top_n": int(top_key.replace("top", "")),
                        "fold": item["fold"]["fold_index"],
                        **metrics,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return []
        grouped = df.groupby(["model", "target", "top_n"], dropna=False)
        summary = []
        for (model, target, top_n), group in grouped:
            positive_rate = float((group["mean_net_return"] > 0).mean())
            avg_return = float(group["mean_net_return"].mean())
            avg_drawdown = float(group["max_drawdown"].mean())
            avg_ratio = group["return_drawdown_ratio"].replace([np.inf, -np.inf], np.nan).mean()
            verdict = "usable" if positive_rate >= 0.7 and avg_return > 0 else "observe" if positive_rate >= 0.5 or avg_return > 0 else "rejected"
            summary.append(
                {
                    "model": model,
                    "target": target,
                    "top_n": int(top_n),
                    "folds": int(len(group)),
                    "positive_fold_rate": round(positive_rate, 6),
                    "avg_mean_net_return": round(avg_return, 6),
                    "avg_win_rate": round(float(group["win_rate"].mean()), 6),
                    "avg_max_drawdown": round(avg_drawdown, 6),
                    "avg_return_drawdown_ratio": round(float(avg_ratio), 6) if pd.notna(avg_ratio) else None,
                    "verdict": verdict,
                }
            )
        MlResearchHarness._apply_topn_consistency_flags(summary)
        return sorted(summary, key=lambda row: (row["verdict"] != "usable", row["target"], -row["positive_fold_rate"], -row["avg_return_drawdown_ratio"] if row["avg_return_drawdown_ratio"] is not None else -999))

    @staticmethod
    def summarize_regime_metrics(results: list[dict]) -> list[dict]:
        rows = []
        for item in results:
            if item.get("status") != "trained":
                continue
            for top_key, regime_metrics in (item.get("regime_metrics") or {}).items():
                for regime, metrics in (regime_metrics or {}).items():
                    if not metrics or not metrics.get("days"):
                        continue
                    rows.append(
                        {
                            "model": item["model"],
                            "target": item["target"],
                            "top_n": int(top_key.replace("top", "")),
                            "fold": item["fold"]["fold_index"],
                            "evaluation_regime": regime,
                            **metrics,
                        }
                    )
        df = pd.DataFrame(rows)
        if df.empty:
            return []
        summary = []
        grouped = df.groupby(["model", "target", "top_n", "evaluation_regime"], dropna=False)
        for (model, target, top_n, regime), group in grouped:
            summary.append(
                {
                    "model": model,
                    "target": target,
                    "top_n": int(top_n),
                    "evaluation_regime": str(regime),
                    "folds_with_days": int(len(group)),
                    "total_days": int(group["days"].sum()),
                    "positive_fold_rate": round(float((group["mean_net_return"] > 0).mean()), 6),
                    "avg_mean_net_return": round(float(group["mean_net_return"].mean()), 6),
                    "avg_win_rate": round(float(group["win_rate"].mean()), 6),
                    "avg_positive_day_rate": round(float(group["positive_day_rate"].mean()), 6),
                    "avg_max_drawdown": round(float(group["max_drawdown"].mean()), 6),
                }
            )
        return sorted(summary, key=lambda row: (row["model"], row["target"], row["top_n"], row["evaluation_regime"]))

    @staticmethod
    def identify_failure_windows(results: list[dict], top_n: int = 10, max_items: int = 50) -> list[dict]:
        failures = []
        top_key = f"top{top_n}"
        for item in results:
            if item.get("status") != "trained":
                continue
            metrics = (item.get("metrics") or {}).get(top_key) or {}
            mean_return = metrics.get("mean_net_return")
            drawdown = metrics.get("max_drawdown")
            if mean_return is None and drawdown is None:
                continue
            severe_return = mean_return is not None and float(mean_return) <= -0.01
            severe_drawdown = drawdown is not None and float(drawdown) <= -0.20
            if not severe_return and not severe_drawdown:
                continue
            fold = item.get("fold") or {}
            failures.append(
                {
                    "model": item["model"],
                    "target": item["target"],
                    "top_n": top_n,
                    "fold": fold.get("fold_index"),
                    "validation_start": fold.get("validation_start"),
                    "validation_end": fold.get("validation_end"),
                    "mean_net_return": mean_return,
                    "max_drawdown": drawdown,
                    "win_rate": metrics.get("win_rate"),
                    "reasons": [
                        reason
                        for reason, active in {
                            "mean_net_return_lte_-1pct": severe_return,
                            "max_drawdown_lte_-20pct": severe_drawdown,
                        }.items()
                        if active
                    ],
                }
            )
        return sorted(failures, key=lambda row: (row["mean_net_return"] or 0, row["max_drawdown"] or 0))[:max_items]

    @staticmethod
    def _apply_topn_consistency_flags(summary: list[dict]) -> None:
        for row in summary:
            row["verdict_is_provisional"] = True
            row["verdict_policy"] = MlResearchHarness.VERDICT_POLICY
            row["risk_tags"] = []
        by_key: dict[tuple[str, str], list[dict]] = {}
        for row in summary:
            by_key.setdefault((row["model"], row["target"]), []).append(row)
        for rows in by_key.values():
            by_top = {row["top_n"]: row for row in rows}
            top10 = by_top.get(10)
            if not top10 or (top10.get("avg_mean_net_return") or 0) <= 0:
                continue
            weak_broader = [
                row["top_n"]
                for top_n, row in by_top.items()
                if top_n > 10 and ((row.get("avg_mean_net_return") or 0) < 0 or (row.get("positive_fold_rate") or 0) < 0.5)
            ]
            if weak_broader:
                for row in rows:
                    row["risk_tags"].append(f"top10_positive_but_broader_topn_weak:{','.join(map(str, sorted(weak_broader)))}")

    @staticmethod
    def _best_summary_row(rows: list[dict] | None) -> dict | None:
        if not rows:
            return None
        filtered = [row for row in rows if isinstance(row, dict)]
        if not filtered:
            return None
        return max(
            filtered,
            key=lambda row: (
                row.get("top_n") == 10,
                row.get("avg_mean_net_return") or row.get("avg_top50_return") or -999,
                row.get("positive_fold_rate") or row.get("positive_window_rate") or 0,
                row.get("avg_return_drawdown_ratio") or -999,
            ),
        )

    @staticmethod
    def summarize_diagnostics(results: list[dict]) -> dict:
        feature_rows = []
        permutation_rows = []
        for item in results:
            if item.get("status") != "trained":
                continue
            fold = item.get("fold") or {}
            diagnostics = item.get("diagnostics") or {}
            for row in diagnostics.get("feature_importance") or []:
                feature_rows.append({"model": item.get("model"), "target": item.get("target"), "fold": fold.get("fold_index"), **row})
            for row in diagnostics.get("permutation_importance") or []:
                permutation_rows.append({"model": item.get("model"), "target": item.get("target"), "fold": fold.get("fold_index"), **row})

        return {
            "feature_importance": MlResearchHarness._summarize_importance_rows(feature_rows),
            "permutation_importance": MlResearchHarness._summarize_importance_rows(permutation_rows),
        }

    @staticmethod
    def _summarize_importance_rows(rows: list[dict]) -> dict:
        if not rows:
            return {"rows": 0, "by_model_target": [], "overall_top_features": []}
        frame = pd.DataFrame(rows)
        if frame.empty or "feature" not in frame.columns:
            return {"rows": 0, "by_model_target": [], "overall_top_features": []}
        value_col = "abs_importance" if "abs_importance" in frame.columns else "importance"
        if value_col not in frame.columns:
            return {"rows": int(len(frame)), "by_model_target": [], "overall_top_features": []}

        by_model_target = []
        for (model, target), group in frame.groupby(["model", "target"], dropna=False):
            importance = (
                group.groupby("feature", dropna=False)[value_col]
                .mean()
                .sort_values(ascending=False)
                .head(DEFAULT_IMPORTANCE_TOP_N)
            )
            by_model_target.append(
                {
                    "model": model,
                    "target": target,
                    "rows": int(len(group)),
                    "top_features": [
                        {"feature": feature, "importance": round(float(score), 6)}
                        for feature, score in importance.items()
                    ],
                }
            )

        overall = (
            frame.groupby("feature", dropna=False)[value_col]
            .mean()
            .sort_values(ascending=False)
            .head(DEFAULT_IMPORTANCE_TOP_N)
        )
        return {
            "rows": int(len(frame)),
            "by_model_target": sorted(by_model_target, key=lambda row: (row["model"], row["target"])),
            "overall_top_features": [
                {"feature": feature, "importance": round(float(score), 6)}
                for feature, score in overall.items()
            ],
        }

    @staticmethod
    def fold_diagnostics(
        model: Any,
        validation_frame: pd.DataFrame,
        feature_cols: list[str],
        target_name: str,
        config: WalkForwardConfig,
        *,
        include_permutation_importance: bool = False,
    ) -> dict:
        x_valid = MlResearchHarness.clean_features(validation_frame, feature_cols)
        feature_importance = MlResearchHarness.model_feature_importance(model, feature_cols, limit=config.diagnostic_top_n)
        permutation_importance = []
        permutation_meta: dict[str, Any] = {
            "status": "skipped",
            "reason": "not_requested",
            "sample_rows": 0,
            "repeats": config.diagnostic_repeats,
            "scoring": None,
        }
        if include_permutation_importance and not x_valid.empty:
            permutation_importance, permutation_meta = MlResearchHarness.model_permutation_importance(
                model,
                validation_frame,
                x_valid,
                feature_cols,
                target_name,
                sample_rows=config.diagnostic_sample_rows,
                repeats=config.diagnostic_repeats,
                random_state=config.random_state,
                limit=config.diagnostic_top_n,
            )
        return {
            "feature_importance": feature_importance,
            "permutation_importance": permutation_importance,
            "permutation_importance_meta": permutation_meta,
        }

    @staticmethod
    def model_feature_importance(model: Any, feature_cols: list[str], limit: int = DEFAULT_IMPORTANCE_TOP_N) -> list[dict]:
        if not hasattr(model, "feature_importances_"):
            return []
        raw = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        if raw.size == 0:
            return []
        if len(feature_cols) != len(raw):
            feature_cols = list(feature_cols[: len(raw)])
        total = float(np.abs(raw).sum())
        rows = []
        for feature, score in zip(feature_cols, raw, strict=False):
            importance = float(abs(score))
            rows.append({"feature": feature, "importance": round(float(score), 6), "abs_importance": round(importance, 6)})
        rows = sorted(rows, key=lambda item: item["abs_importance"], reverse=True)
        if total > 0:
            for row in rows:
                row["importance_share"] = round(row["abs_importance"] / total, 6)
        return rows[:limit]

    @staticmethod
    def model_permutation_importance(
        model: Any,
        validation_frame: pd.DataFrame,
        x_valid: pd.DataFrame,
        feature_cols: list[str],
        target_name: str,
        *,
        sample_rows: int = DEFAULT_IMPORTANCE_SAMPLE_ROWS,
        repeats: int = DEFAULT_IMPORTANCE_REPEATS,
        random_state: int = 42,
        limit: int = DEFAULT_IMPORTANCE_TOP_N,
    ) -> tuple[list[dict], dict]:
        if x_valid.empty:
            return [], {"status": "skipped", "reason": "empty_validation_frame", "sample_rows": 0, "repeats": repeats, "scoring": None}
        y_valid = MlResearchHarness.target_values(validation_frame.loc[x_valid.index], target_name)
        keep = y_valid.notna()
        x_valid = x_valid.loc[keep]
        y_valid = y_valid.loc[keep]
        if x_valid.empty or y_valid.empty:
            return [], {"status": "skipped", "reason": "empty_target_frame", "sample_rows": 0, "repeats": repeats, "scoring": None}

        sample_size = min(sample_rows, len(x_valid))
        if sample_size <= 0:
            return [], {"status": "skipped", "reason": "invalid_sample_size", "sample_rows": 0, "repeats": repeats, "scoring": None}

        sampled = x_valid.sample(n=sample_size, random_state=random_state) if len(x_valid) > sample_size else x_valid
        sampled_y = y_valid.loc[sampled.index]
        scoring = "roc_auc" if target_name == "stable" else "neg_mean_absolute_error"
        try:
            from sklearn.inspection import permutation_importance
        except Exception as exc:
            return [], {"status": "skipped", "reason": f"sklearn_unavailable:{exc}", "sample_rows": int(len(sampled)), "repeats": repeats, "scoring": scoring}

        try:
            result = permutation_importance(
                model,
                sampled,
                sampled_y,
                scoring=scoring,
                n_repeats=repeats,
                random_state=random_state,
                n_jobs=1,
            )
        except Exception as exc:
            return [], {"status": "skipped", "reason": f"permutation_failed:{exc}", "sample_rows": int(len(sampled)), "repeats": repeats, "scoring": scoring}

        rows = []
        for feature, score, std in sorted(
            zip(feature_cols, result.importances_mean, result.importances_std, strict=False),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:limit]:
            rows.append(
                {
                    "feature": feature,
                    "importance": round(float(score), 6),
                    "abs_importance": round(abs(float(score)), 6),
                    "importance_std": round(float(std), 6),
                }
            )

        return rows, {
            "status": "pass",
            "sample_rows": int(len(sampled)),
            "repeats": repeats,
            "scoring": scoring,
        }
