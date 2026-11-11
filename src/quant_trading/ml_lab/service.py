from __future__ import annotations



from dataclasses import dataclass

from datetime import datetime

import importlib.util

from pathlib import Path

import csv

import json

from typing import Any



import pandas as pd

from sqlalchemy import and_, func, select

from sqlalchemy.dialects.postgresql import insert



from ..db import DailyQuote, FactorSnapshot, MlBacktestRun, MlDualScore, MlExperimentArtifact, MlExperimentRun, MlTopCandidate, Stock, session_scope

from ..factor_lab.service import FactorLabConfig, FactorLabError, FactorLabService

from ..factor_lab.batch import VALIDATION_WINDOWS

from ..factor_lab.metrics import safe_float
from .feature_policy import is_optional_feature





MODEL_VERSION = "dual_score_rule_v1"
RESEARCH_GATE_STATUSES = {"rejected", "observe_only", "candidate", "usable"}

TOP_N = 10
FACTOR_KEEP = "\u4fdd\u7559"
FACTOR_WATCH = "\u89c2\u5bdf"
FACTOR_DOWNGRADE = "\u964d\u7ea7"





STABLE_WEIGHTS = {

    "relative_industry_strength_20d": 0.18,

    "main_net_to_circ_mv": 0.16,

    "main_net_to_amount": 0.14,

    "turnover_rate": 0.10,

    "return_20d": 0.10,

    "distance_to_20d_high": 0.10,

    "volatility_20d": 0.08,

    "amplitude_20d": 0.06,

    "pb": 0.04,

    "listing_days": 0.04,

}



AGGRESSIVE_WEIGHTS = {

    "return_5d": 0.18,

    "return_20d": 0.16,

    "volume_ratio": 0.14,

    "main_net_to_amount": 0.14,

    "main_net_to_circ_mv": 0.12,

    "relative_industry_strength_20d": 0.10,

    "turnover_rate": 0.08,

    "distance_to_20d_high": 0.04,

    "amplitude_20d": 0.04,

}



RISK_THRESHOLDS = {

    "low_liquidity_amount": 30000.0,

    "short_listing_days": 120.0,

    "high_volatility_20d": 0.045,

    "high_amplitude_20d": 0.25,

    "near_limit_up_pct": 9.0,

    "recent_surge_5d": 0.18,

}



VALIDATION_WINDOWS = [

    {"name": "bull_start", "start_date": "20250601", "end_date": "20250731"},

    {"name": "bull_mid", "start_date": "20260105", "end_date": "20260227"},

    {"name": "bull_late", "start_date": "20260401", "end_date": "20260520"},

    {"name": "control_choppy", "start_date": "20260301", "end_date": "20260415"},

]





class MlLabError(RuntimeError):

    """Raised when ML Lab research tasks cannot be completed from real local data."""





@dataclass(frozen=True)

class MlLabConfig:

    model_version: str = MODEL_VERSION

    factor_set: str = "factor_lab_v1"

    feature_set: str | None = None

    buy_fee_rate: float = 0.0003

    sell_fee_rate: float = 0.0003

    stamp_tax_rate: float = 0.001

    slippage: float = 0.0005

    board_lot: int = 100





@dataclass

class MlLabService:

    config: MlLabConfig = MlLabConfig()

    @property
    def active_factor_set(self) -> str:
        return self.config.feature_set or self.config.factor_set



    def readiness(self, start_date: str = "20240101", end_date: str = "20260601") -> dict:

        with session_scope() as session:

            snapshot_days = (

                session.query(func.count(func.distinct(FactorSnapshot.trade_date)))

                .filter(

                    FactorSnapshot.factor_set == self.active_factor_set,

                    FactorSnapshot.trade_date >= start_date,

                    FactorSnapshot.trade_date <= end_date,

                )

                .scalar()

                or 0

            )

            quote_days = (

                session.query(func.count(func.distinct(DailyQuote.trade_date)))

                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)

                .scalar()

                or 0

            )

            score_rows = (

                session.query(func.count())

                .select_from(MlDualScore)

                .filter(

                    MlDualScore.model_version == self.config.model_version,

                    MlDualScore.trade_date >= start_date,

                    MlDualScore.trade_date <= end_date,

                )

                .scalar()

                or 0

            )

        return {

            "task": "ml_lab_readiness",

            "model_version": self.config.model_version,

            "factor_set": self.active_factor_set,

            "start_date": start_date,

            "end_date": end_date,

            "snapshot_days": int(snapshot_days),

            "quote_days": int(quote_days),

            "score_rows": int(score_rows),

            "model_dependencies": self.model_dependencies(),

        }



    def score_date(self, trade_date: str) -> dict:

        df = self._score_frame(trade_date)

        if df.empty:

            raise MlLabError(f"No factor snapshots found for trade_date={trade_date}. Build Factor Lab snapshots first.")



        ranked = self._rank_scores(df)

        self._persist_scores(ranked)

        top_rows = self._persist_top_candidates(ranked)

        self._persist_experiment_run(

            experiment_type="score_date",

            start_date=trade_date,

            end_date=trade_date,

            summary={"rows": int(len(ranked)), "top_candidates": top_rows},

        )

        return {

            "task": "ml_lab_score_date",

            "trade_date": trade_date,

            "model_version": self.config.model_version,

            "rows": int(len(ranked)),

            "top_candidates": top_rows,

        }



    def build_missing_snapshots(self, start_date: str, end_date: str, limit: int | None = None, force: bool = False) -> dict:

        factor_service = FactorLabService(FactorLabConfig(factor_set=self.active_factor_set))

        trade_dates = self._trade_dates(start_date, end_date)

        results = []

        built = skipped = failed = 0

        for trade_date in trade_dates:

            if limit is not None and built >= limit:

                break

            if not force and self._has_snapshot(trade_date):

                skipped += 1

                continue

            try:

                result = factor_service.build_snapshot(trade_date)

                results.append(result)

                built += 1

            except FactorLabError as exc:

                results.append({"trade_date": trade_date, "error": str(exc)})

                failed += 1

        return {

            "task": "build_missing_snapshots",

            "start_date": start_date,

            "end_date": end_date,

            "built": built,

            "skipped": skipped,

            "failed": failed,

            "force": force,

            "results": results,

        }



    def get_top10(self, trade_date: str) -> dict:

        with session_scope() as session:

            rows = (

                session.query(MlTopCandidate)

                .filter(MlTopCandidate.trade_date == trade_date, MlTopCandidate.model_version == self.config.model_version)

                .order_by(MlTopCandidate.list_type.asc(), MlTopCandidate.rank.asc())

                .all()

            )

        if not rows:

            self.score_date(trade_date)

            return self.get_top10(trade_date)

        result = {"stable": [], "aggressive": []}

        for row in rows:

            result[row.list_type].append(

                {

                    "rank": row.rank,

                    "ts_code": row.ts_code,

                    "name": row.name,

                    "stable_score": row.stable_score,

                    "aggressive_score": row.aggressive_score,

                    "contribution_factors": row.contribution_factors or [],

                    "risk_tags": row.risk_tags or [],

                }

                )

        return {"trade_date": trade_date, "model_version": self.config.model_version, "top10": result}



    def build_top10(self, trade_date: str) -> dict:

        """Backward-compatible alias for the research Top10 view."""

        return self.get_top10(trade_date)



    def backtest(

        self,

        start_date: str,

        end_date: str,

        stable_count: int = 5,

        aggressive_count: int = 2,

    ) -> dict:

        trade_dates = self._trade_dates(start_date, end_date)

        if len(trade_dates) < 3:

            raise MlLabError("Backtest requires at least three trading days.")



        trades: list[dict[str, Any]] = []

        periods = 0

        for index in range(0, len(trade_dates) - 2):

            signal_date = trade_dates[index]

            buy_date = trade_dates[index + 1]

            sell_date = trade_dates[index + 2]

            self._ensure_scores(signal_date)

            picks = self._portfolio_picks(signal_date, stable_count=stable_count, aggressive_count=aggressive_count)

            if not picks:

                continue

            periods += 1

            for pick in picks:

                trade = self._simulate_one_lot_trade(pick, signal_date, buy_date, sell_date, end_date)

                if trade:

                    trades.append(trade)



        summary = self._backtest_summary(trades, periods, start_date, end_date, stable_count, aggressive_count)

        self._persist_backtest(summary)

        self._persist_experiment_run(

            experiment_type="backtest",

            start_date=start_date,

            end_date=end_date,

            summary=summary,

        )

        return summary



    def export_training_frame(self, start_date: str, end_date: str, output_path: str | Path, trade_dates: list[str] | None = None) -> dict:

        selected_trade_dates = trade_dates or self._trade_dates(start_date, end_date)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_names = self.feature_names(self.active_factor_set)
        feature_manifest = self.build_feature_manifest(feature_names)
        label_columns = [
            "label_buy_date",
            "label_sell_date",
            "label_tradable",
            "label_net_return",
            "label_stable_positive",
            "label_top5",
            "label_top10",
        ]
        fieldnames = sorted(
            {
                "trade_date",
                "ts_code",
                "stable_score",
                "aggressive_score",
                "stable_rank",
                "aggressive_rank",
                *[f"feature_{name}" for name in feature_names],
                *label_columns,
            }
        )
        rows_written = 0
        f = path.open("w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trade_date in selected_trade_dates:

            scored = self._rank_scores(self._score_frame(trade_date))

            if scored.empty:

                continue

            labels = self._labels_for_date(trade_date)

            for item in scored.to_dict("records"):

                label = {column: None for column in label_columns}
                label["label_tradable"] = False
                label.update(labels.get(item["ts_code"], {}))
                features = item.get("features") or {}

                writer.writerow(

                    {

                        "trade_date": trade_date,

                        "ts_code": item["ts_code"],

                        "stable_score": item["stable_score"],

                        "aggressive_score": item["aggressive_score"],

                        "stable_rank": item["stable_rank"],

                        "aggressive_rank": item["aggressive_rank"],

                        **{f"feature_{name}": features.get(name) for name in feature_names},

                        **label,

                    }

                )
                rows_written += 1



        f.close()

        run_id = self._persist_experiment_run(

            experiment_type="export_training_frame",

            start_date=start_date,

            end_date=end_date,

            summary={
                "rows": rows_written,
                "output_path": str(path),
                "trade_date_count": len(selected_trade_dates),
                "feature_set": self.active_factor_set,
                "feature_count": len(feature_names),
            },

            artifact_type="training_frame",

            artifact_path=str(path),

        )

        return {
            "task": "export_training_frame",
            "output_path": str(path),
            "rows": rows_written,
            "trade_date_count": len(selected_trade_dates),
            "experiment_run_id": run_id,
            "feature_set": self.active_factor_set,
            "feature_contract": feature_manifest,
        }



    def train_models(self, training_frame_path: str | Path, output_dir: str | Path = "reports/ml_lab/models", factor_pool_version: str | None = None) -> dict:

        df = pd.read_csv(training_frame_path)

        if df.empty:

            raise MlLabError("Training frame is empty.")

        feature_cols = sorted([col for col in df.columns if col.startswith("feature_")])
        feature_gate = self.feature_missingness_gate(df, feature_cols)
        if feature_gate["status"] != "pass":
            raise MlLabError(f"Training frame feature missingness gate failed: {feature_gate['blocking_features']}")

        required = {"label_stable_positive", "label_top5", "label_top10", "label_tradable", "trade_date"}

        missing = sorted(required - set(df.columns))

        if missing:

            raise MlLabError(f"Training frame missing required label columns: {missing}")



        working = df[df["label_tradable"].astype(str).str.lower().isin({"true", "1"})].copy()

        working[feature_cols] = working[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        train_df = working[~working["trade_date"].astype(str).map(self._is_validation_date)].copy()

        if train_df.empty:

            raise MlLabError("No training rows remain after excluding validation windows.")



        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        report = {

            "task": "train_ml_lab_models",

            "training_frame_path": str(training_frame_path),
            "factor_pool_version": factor_pool_version,

            "feature_count": len(feature_cols),

            "feature_missingness_gate": feature_gate,

            "train_rows": int(len(train_df)),

            "validation_rows": int(len(working) - len(train_df)),

            "models": [],

        }

        for model_name in ["lightgbm", "xgboost", "catboost"]:

            for target_name, label_col in [("stable", "label_stable_positive"), ("aggressive", "label_top10")]:

                item = self._train_one_model(model_name, target_name, label_col, feature_cols, train_df, working, output_path)

                report["models"].append(item)



        report_path = output_path / "model_comparison_report.json"
        summary_path = output_path / "model_validation_summary.json"

        validation_summary = self._model_validation_summary(report["models"])
        report["validation_summary"] = validation_summary
        summary_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        ledger_path = self._write_experiment_ledger(output_path, report_path, summary_path, report_path)

        run_id = self._persist_experiment_run(

            experiment_type="train_models",

            start_date=str(train_df["trade_date"].min()),

            end_date=str(train_df["trade_date"].max()),

            summary={"feature_count": len(feature_cols), "train_rows": int(len(train_df)), "validation_rows": int(len(working) - len(train_df)), "report_path": str(report_path)},

            artifact_type="model_report",

            artifact_path=str(report_path),

        )

        return {**report, "report_path": str(report_path), "experiment_run_id": run_id}



    def build_factor_convergence_report(

        self,

        training_frame_path: str | Path = "reports/ml_lab/training_frame_full.csv",

        output_dir: str | Path = "reports/ml_lab/convergence",

        model_report_path: str | Path = "reports/ml_lab/models_full/model_comparison_report.json",

        factor_pool_version: str | None = None,

    ) -> dict:

        training_path = Path(training_frame_path)

        report_path = Path(model_report_path)

        if not training_path.exists():

            raise MlLabError(f"Training frame not found: {training_path}")

        if not report_path.exists():

            raise MlLabError(f"Model report not found: {report_path}")



        model_report = json.loads(report_path.read_text(encoding="utf-8"))

        convergence_rows = self._convergence_rows(model_report.get("models", []))

        factor_pool_rows = self._factor_pool_rows(convergence_rows)



        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        summary_path = output_path / "factor_convergence_summary.csv"

        detail_path = output_path / "factor_convergence_details.json"

        pool_path = output_path / "factor_pool_convergence.csv"

        self._write_csv(summary_path, convergence_rows)

        self._write_csv(pool_path, factor_pool_rows)

        detail_payload = {

            "task": "factor_convergence_report",

            "training_frame_path": str(training_path),

            "model_report_path": str(report_path),
            "factor_pool_version": factor_pool_version,

            "summary": convergence_rows,

            "factor_pool": factor_pool_rows,

            "models": model_report.get("models", []),

        }

        detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        run_id = self._persist_experiment_run(

            experiment_type="factor_convergence",

            start_date=None,

            end_date=None,

            summary={"summary_path": str(summary_path), "detail_path": str(detail_path), "pool_path": str(pool_path), "rows": len(convergence_rows)},

            artifact_type="convergence_report",

            artifact_path=str(detail_path),

        )
        ledger_path = self._write_experiment_ledger(output_path, report_path, summary_path, detail_path)

        return {

            "task": "factor_convergence_report",

            "summary_path": str(summary_path),

            "detail_path": str(detail_path),

            "pool_path": str(pool_path),

            "rows": convergence_rows,

            "factor_pool": factor_pool_rows,

            "experiment_run_id": run_id,

        }



    @staticmethod

    def _convergence_rows(models: list[dict]) -> list[dict]:
        rows = []
        for model in models:
            target = model.get("target")
            factor_name = "stable_score" if target == "stable" else "aggressive_score"
            evaluations = model.get("evaluations", [])
            window_count = len(evaluations)
            positive_windows = 0
            top50_sum = 0.0
            label_rate_sum = 0.0
            for row in evaluations:
                top50 = safe_float((row.get("top50") or {}).get("mean_net_return"))
                label_rate = safe_float((row.get("top10") or {}).get("label_rate"))
                if top50 is not None and top50 > 0:
                    positive_windows += 1
                top50_sum += top50 or 0.0
                label_rate_sum += label_rate or 0.0
            stability = positive_windows / window_count if window_count else 0.0
            avg_top50 = top50_sum / window_count if window_count else 0.0
            avg_label_rate = label_rate_sum / window_count if window_count else 0.0
            if stability >= 0.75 and avg_top50 > 0:
                verdict = FACTOR_KEEP
            elif stability >= 0.5 or avg_top50 > 0:
                verdict = FACTOR_WATCH
            else:
                verdict = FACTOR_DOWNGRADE
            rows.append({
                "factor": factor_name,
                "verdict": verdict,
                "window_count": window_count,
                "positive_window_rate": round(stability, 6),
                "avg_top50_return": round(avg_top50, 6),
                "avg_top10_label_rate": round(avg_label_rate, 6),
            })
        verdict_order = {FACTOR_KEEP: 0, FACTOR_WATCH: 1, FACTOR_DOWNGRADE: 2}
        return sorted(rows, key=lambda row: (verdict_order.get(row["verdict"], 99), -row["positive_window_rate"], -row["avg_top50_return"]))

    @staticmethod
    def _factor_pool_rows(convergence_rows: list[dict]) -> list[dict]:
        rows = []
        for item in convergence_rows:
            verdict = item.get("verdict")
            status = verdict if verdict in {FACTOR_KEEP, FACTOR_WATCH, FACTOR_DOWNGRADE} else FACTOR_DOWNGRADE
            rows.append({**item, "status": status})
        return rows

    def _score_frame(self, trade_date: str) -> pd.DataFrame:

        with session_scope() as session:

            rows = (

                session.query(FactorSnapshot, Stock)

                .join(Stock, Stock.ts_code == FactorSnapshot.ts_code)

                .filter(FactorSnapshot.trade_date == trade_date, FactorSnapshot.factor_set == self.active_factor_set)

                .all()

            )

        records = []

        for snapshot, stock in rows:

            raw = snapshot.raw_factors or {}

            normalized = snapshot.normalized_factors or {}

            feature_names = self.feature_names(self.active_factor_set)

            features = {name: safe_float(raw.get(name)) for name in feature_names}

            normalized_features = {name: safe_float(normalized.get(name)) for name in feature_names}

            data_quality = snapshot.data_quality or {}

            records.append(

                {

                    "trade_date": snapshot.trade_date,

                    "ts_code": snapshot.ts_code,

                    "name": stock.name,

                    "industry": snapshot.industry,

                    "raw_features": features,

                    "features": normalized_features,

                    "data_quality": data_quality,

                    "pct_chg": features.get("pct_chg"),

                }

            )

        return pd.DataFrame(records)



    def _labels_for_date(self, trade_date: str) -> dict[str, dict]:

        buy_date, sell_date = self._next_two_trade_dates(trade_date)

        if not buy_date or not sell_date:

            return {}

        with session_scope() as session:

            rows = (

                session.query(DailyQuote.ts_code, DailyQuote.open, DailyQuote.pct_chg, DailyQuote.close, DailyQuote.trade_date)

                .filter(DailyQuote.trade_date.in_([buy_date, sell_date]))

                .all()

            )

        by_stock: dict[str, dict[str, Any]] = {}

        for ts_code, open_price, pct_chg, close_price, row_date in rows:

            by_stock.setdefault(ts_code, {})[row_date] = {

                "open": safe_float(open_price),

                "pct_chg": safe_float(pct_chg),

                "close": safe_float(close_price),

            }



        labels = {}

        returns = []

        cost = self.config.buy_fee_rate + self.config.sell_fee_rate + self.config.stamp_tax_rate + self.config.slippage * 2

        for ts_code, quotes in by_stock.items():

            buy = quotes.get(buy_date)

            sell = quotes.get(sell_date)

            tradable = bool(

                buy

                and sell

                and buy.get("open") not in (None, 0)

                and sell.get("close") not in (None, 0)

                and (buy.get("pct_chg") is None or buy.get("pct_chg") < 9.8)

                and (sell.get("pct_chg") is None or sell.get("pct_chg") > -9.8)

            )

            if tradable:

                net = float(sell["close"]) / float(buy["open"]) - 1.0 - cost

                returns.append(net)

            else:

                net = None

            labels[ts_code] = {

                "label_tradable": tradable,

                "label_net_return": net,

                "label_stable_positive": int(net > 0) if net is not None else None,

                "label_buy_date": buy_date,

                "label_sell_date": sell_date,

            }



        clean = pd.Series([item["label_net_return"] for item in labels.values() if item["label_net_return"] is not None])

        top5_cut = clean.quantile(0.95) if not clean.empty else None

        top10_cut = clean.quantile(0.90) if not clean.empty else None

        for item in labels.values():

            net = item["label_net_return"]

            item["label_top5"] = int(net >= top5_cut) if net is not None and top5_cut is not None else None

            item["label_top10"] = int(net >= top10_cut) if net is not None and top10_cut is not None else None

        return labels



    def _rank_scores(self, df: pd.DataFrame) -> pd.DataFrame:

        if df.empty:

            return df

        records = []

        for row in df.to_dict("records"):

            features = row.get("features") or {}

            raw = row.get("raw_features") or {}

            data_quality = row.get("data_quality") or {}

            stable = self._weighted_score(features, STABLE_WEIGHTS)

            aggressive = self._weighted_score(features, AGGRESSIVE_WEIGHTS)

            stable = self._apply_stable_penalty(stable, raw, data_quality)

            aggressive = self._apply_aggressive_penalty(aggressive, raw, data_quality)

            records.append({**row, "stable_score": stable, "aggressive_score": aggressive})

        result = pd.DataFrame(records)

        result["stable_rank"] = result["stable_score"].rank(method="first", ascending=False).astype(int)

        result["aggressive_rank"] = result["aggressive_score"].rank(method="first", ascending=False).astype(int)

        return result.sort_values(["stable_rank", "aggressive_rank"]).reset_index(drop=True)



    def _persist_scores(self, df: pd.DataFrame) -> None:

        now = datetime.utcnow()

        with session_scope() as session:

            for row in df.to_dict("records"):

                stmt = insert(MlDualScore).values(

                    trade_date=row["trade_date"],

                    ts_code=row["ts_code"],

                    model_version=self.config.model_version,

                    stable_score=round(float(row["stable_score"]), 6),

                    aggressive_score=round(float(row["aggressive_score"]), 6),

                    stable_rank=int(row["stable_rank"]),

                    aggressive_rank=int(row["aggressive_rank"]),

                    created_at=now,

                )

                stmt = stmt.on_conflict_do_update(

                    constraint="uq_ml_dual_score_date_stock_model",

                    set_={

                        "stable_score": stmt.excluded.stable_score,

                        "aggressive_score": stmt.excluded.aggressive_score,

                        "stable_rank": stmt.excluded.stable_rank,

                        "aggressive_rank": stmt.excluded.aggressive_rank,

                        "created_at": now,

                    },

                )

                session.execute(stmt)



    def _persist_top_candidates(self, df: pd.DataFrame) -> dict[str, int]:

        now = datetime.utcnow()

        stable = df.sort_values("stable_rank").head(TOP_N)

        aggressive = df.sort_values("aggressive_rank").head(TOP_N)

        persisted = {"stable": 0, "aggressive": 0}

        with session_scope() as session:

            for list_type, rows in [("stable", stable), ("aggressive", aggressive)]:

                rank_col = f"{list_type}_rank"

                for row in rows.to_dict("records"):

                    weights = STABLE_WEIGHTS if list_type == "stable" else AGGRESSIVE_WEIGHTS

                    stmt = insert(MlTopCandidate).values(

                        trade_date=row["trade_date"],

                        ts_code=row["ts_code"],

                        name=row.get("name"),

                        list_type=list_type,

                        model_version=self.config.model_version,

                        rank=int(row[rank_col]),

                        stable_score=round(float(row["stable_score"]), 6),

                        aggressive_score=round(float(row["aggressive_score"]), 6),

                        contribution_factors=self._contribution_factors(row.get("features") or {}, weights),

                        risk_tags=self._risk_tags(row.get("raw_features") or {}, row.get("data_quality") or {}),

                        created_at=now,

                    )

                    stmt = stmt.on_conflict_do_update(

                        constraint="uq_ml_top_candidate",

                        set_={

                            "name": stmt.excluded.name,

                            "rank": stmt.excluded.rank,

                            "stable_score": stmt.excluded.stable_score,

                            "aggressive_score": stmt.excluded.aggressive_score,

                            "contribution_factors": stmt.excluded.contribution_factors,

                            "risk_tags": stmt.excluded.risk_tags,

                            "created_at": now,

                        },

                    )

                    session.execute(stmt)

                    persisted[list_type] += 1

        return persisted



    def _portfolio_picks(self, trade_date: str, stable_count: int, aggressive_count: int) -> list[dict]:

        with session_scope() as session:

            stable = (

                session.query(MlDualScore, Stock)

                .join(Stock, Stock.ts_code == MlDualScore.ts_code)

                .filter(MlDualScore.trade_date == trade_date, MlDualScore.model_version == self.config.model_version)

                .order_by(MlDualScore.stable_rank.asc())

                .limit(TOP_N + stable_count + aggressive_count)

                .all()

            )

            aggressive = (

                session.query(MlDualScore, Stock)

                .join(Stock, Stock.ts_code == MlDualScore.ts_code)

                .filter(MlDualScore.trade_date == trade_date, MlDualScore.model_version == self.config.model_version)

                .order_by(MlDualScore.aggressive_rank.asc())

                .limit(TOP_N + stable_count + aggressive_count)

                .all()

            )

        picks = []

        used = set()

        for score, stock in stable:

            if len([p for p in picks if p["bucket"] == "stable"]) >= stable_count:

                break

            picks.append(self._pick_payload(score, stock, "stable"))

            used.add(score.ts_code)

        for score, stock in aggressive:

            if len([p for p in picks if p["bucket"] == "aggressive"]) >= aggressive_count:

                break

            if score.ts_code in used:

                continue

            picks.append(self._pick_payload(score, stock, "aggressive"))

            used.add(score.ts_code)

        return picks



    def _simulate_one_lot_trade(self, pick: dict, signal_date: str, buy_date: str, sell_date: str, end_date: str) -> dict | None:

        buy_quote = self._quote(pick["ts_code"], buy_date)

        if not buy_quote or buy_quote.open in (None, 0):

            return None

        if safe_float(buy_quote.pct_chg) is not None and safe_float(buy_quote.pct_chg) >= 9.8:

            return None



        exit_quote = self._tradable_exit_quote(pick["ts_code"], sell_date, end_date)

        if not exit_quote or exit_quote.close in (None, 0):

            return None



        shares = self.config.board_lot

        buy_price = float(buy_quote.open) * (1 + self.config.slippage)

        sell_price = float(exit_quote.close) * (1 - self.config.slippage)

        buy_amount = shares * buy_price

        sell_amount = shares * sell_price

        buy_fee = buy_amount * self.config.buy_fee_rate

        sell_fee = sell_amount * self.config.sell_fee_rate

        stamp_tax = sell_amount * self.config.stamp_tax_rate

        profit = sell_amount - sell_fee - stamp_tax - buy_amount - buy_fee

        return {

            **pick,

            "signal_date": signal_date,

            "buy_date": buy_date,

            "sell_date": exit_quote.trade_date,

            "shares": shares,

            "buy_price": buy_price,

            "sell_price": sell_price,

            "buy_cost": buy_amount + buy_fee,

            "sell_income": sell_amount - sell_fee - stamp_tax,

            "profit": profit,

            "return_on_cost": profit / (buy_amount + buy_fee) if buy_amount else None,

        }



    def _backtest_summary(self, trades: list[dict], periods: int, start_date: str, end_date: str, stable_count: int, aggressive_count: int) -> dict:

        total_profit = sum(safe_float(item.get("profit")) or 0.0 for item in trades)

        total_cost = sum(safe_float(item.get("buy_cost")) or 0.0 for item in trades)

        wins = sum(1 for item in trades if (safe_float(item.get("profit")) or 0.0) > 0)

        stable_trades = [item for item in trades if item.get("bucket") == "stable"]

        aggressive_trades = [item for item in trades if item.get("bucket") == "aggressive"]

        summary = {

            "task": "ml_lab_backtest",

            "model_version": self.config.model_version,

            "start_date": start_date,

            "end_date": end_date,

            "portfolio_rule": f"stable{stable_count}_aggressive{aggressive_count}_one_lot",

            "periods": periods,

            "trades": len(trades),

            "total_profit": round(total_profit, 4),

            "total_cost": round(total_cost, 4),

            "total_return_on_cost": round(total_profit / total_cost, 6) if total_cost else None,

            "win_rate": round(wins / len(trades), 6) if trades else None,

            "stable": self._bucket_summary(stable_trades),

            "aggressive": self._bucket_summary(aggressive_trades),

        }

        return summary



    def _persist_backtest(self, summary: dict) -> None:

        with session_scope() as session:

            session.add(

                MlBacktestRun(

                    model_version=self.config.model_version,

                    start_date=summary["start_date"],

                    end_date=summary["end_date"],

                    portfolio_rule=summary["portfolio_rule"],

                    periods=summary["periods"],

                    trades=summary["trades"],

                    total_profit=summary["total_profit"],

                    total_return_on_cost=summary["total_return_on_cost"],

                    win_rate=summary["win_rate"],

                    summary=summary,

                )

            )



    def _persist_experiment_run(

        self,

        experiment_type: str,

        start_date: str | None,

        end_date: str | None,

        summary: dict | None,

        artifact_type: str | None = None,

        artifact_path: str | None = None,

    ) -> int:

        with session_scope() as session:

            run = MlExperimentRun(

                model_version=self.config.model_version,

                experiment_type=experiment_type,

                start_date=start_date,

                end_date=end_date,

                status="completed",

                summary=summary,

            )

            session.add(run)

            session.flush()

            if artifact_type is not None:

                session.add(

                    MlExperimentArtifact(

                        experiment_run_id=run.id,

                        artifact_type=artifact_type,

                        artifact_path=artifact_path,

                        summary=summary,

                    )

                )

            session.flush()

            return int(run.id)



    def _next_two_trade_dates(self, trade_date: str) -> tuple[str | None, str | None]:

        dates = self._trade_dates(trade_date, "99991231", limit=3)

        if len(dates) < 3:

            return None, None

        return dates[1], dates[2]



    def _ensure_scores(self, trade_date: str) -> None:

        with session_scope() as session:

            exists = (

                session.query(func.count())

                .select_from(MlDualScore)

                .filter(MlDualScore.trade_date == trade_date, MlDualScore.model_version == self.config.model_version)

                .scalar()

                or 0

            )

        if exists == 0:

            self.score_date(trade_date)



    def _has_snapshot(self, trade_date: str) -> bool:

        with session_scope() as session:

            count = (

                session.query(func.count())

                .select_from(FactorSnapshot)

                .filter(FactorSnapshot.trade_date == trade_date, FactorSnapshot.factor_set == self.active_factor_set)

                .scalar()

                or 0

            )

        return count > 0



    def _quote(self, ts_code: str, trade_date: str) -> DailyQuote | None:

        with session_scope() as session:

            return session.query(DailyQuote).filter(DailyQuote.ts_code == ts_code, DailyQuote.trade_date == trade_date).first()



    def _tradable_exit_quote(self, ts_code: str, sell_date: str, end_date: str) -> DailyQuote | None:

        with session_scope() as session:

            rows = (

                session.query(DailyQuote)

                .filter(DailyQuote.ts_code == ts_code, DailyQuote.trade_date >= sell_date, DailyQuote.trade_date <= end_date)

                .order_by(DailyQuote.trade_date.asc())

                .all()

            )

        for row in rows:

            pct = safe_float(row.pct_chg)

            if pct is None or pct > -9.8:

                return row

        return rows[-1] if rows else None



    @staticmethod

    def _weighted_score(features: dict[str, float | None], weights: dict[str, float]) -> float:

        total_weight = 0.0

        total = 0.0

        for name, weight in weights.items():

            value = safe_float(features.get(name))

            if value is None:

                continue

            total += value * weight

            total_weight += weight

        return total / total_weight if total_weight else 0.0



    @staticmethod

    def _apply_stable_penalty(score: float, raw: dict, data_quality: dict) -> float:

        penalty = MlLabService._data_quality_penalty(data_quality)

        vol = safe_float(raw.get("volatility_20d"))

        amp = safe_float(raw.get("amplitude_20d"))

        surge = safe_float(raw.get("return_5d"))

        if vol is not None and vol > RISK_THRESHOLDS["high_volatility_20d"]:

            penalty += 8

        if amp is not None and amp > RISK_THRESHOLDS["high_amplitude_20d"]:

            penalty += 6

        if surge is not None and surge > RISK_THRESHOLDS["recent_surge_5d"]:

            penalty += 5

        return max(0.0, min(100.0, score - penalty))



    @staticmethod

    def _apply_aggressive_penalty(score: float, raw: dict, data_quality: dict) -> float:

        penalty = MlLabService._data_quality_penalty(data_quality) * 0.5

        listing_days = safe_float(raw.get("listing_days"))

        amount = safe_float(raw.get("amount"))

        if listing_days is not None and listing_days < RISK_THRESHOLDS["short_listing_days"]:

            penalty += 8

        if amount is not None and amount < RISK_THRESHOLDS["low_liquidity_amount"]:

            penalty += 8

        return max(0.0, min(100.0, score - penalty))



    @staticmethod

    @staticmethod

    def _write_csv(path: Path, rows: list[dict]) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = sorted({key for row in rows for key in row}) if rows else []

        with path.open("w", encoding="utf-8-sig", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            writer.writerows(rows)



    def _data_quality_penalty(data_quality: dict) -> float:

        constraints = data_quality.get("constraints") or {}

        penalty = 0.0

        if constraints.get("moneyflow_missing"):

            penalty += 12

        if constraints.get("low_liquidity"):

            penalty += 10

        if constraints.get("short_listing"):

            penalty += 6

        return penalty



    @staticmethod

    def _contribution_factors(features: dict[str, float | None], weights: dict[str, float], limit: int = 5) -> list[dict]:

        rows = []

        for name, weight in weights.items():

            value = safe_float(features.get(name))

            if value is None:

                continue

            rows.append({"factor": name, "value": round(value, 4), "weight": weight, "contribution": round(value * weight, 4)})

        return sorted(rows, key=lambda item: item["contribution"], reverse=True)[:limit]



    @staticmethod

    def _risk_tags(raw: dict, data_quality: dict) -> list[str]:

        tags: list[str] = []

        constraints = data_quality.get("constraints") or {}

        if constraints.get("moneyflow_missing"):

            tags.append("资金流缺失")

        if constraints.get("low_liquidity"):

            tags.append("流动性偏低")

        if constraints.get("short_listing"):

            tags.append("上市时间较短")

        if (safe_float(raw.get("volatility_20d")) or 0) > RISK_THRESHOLDS["high_volatility_20d"]:

            tags.append("20日波动偏高")

        if (safe_float(raw.get("amplitude_20d")) or 0) > RISK_THRESHOLDS["high_amplitude_20d"]:

            tags.append("20日振幅偏高")

        if (safe_float(raw.get("return_5d")) or 0) > RISK_THRESHOLDS["recent_surge_5d"]:

            tags.append("短期涨幅过热")

        return tags



    @staticmethod

    def _pick_payload(score: MlDualScore, stock: Stock, bucket: str) -> dict:

        return {

            "bucket": bucket,

            "ts_code": score.ts_code,

            "name": stock.name,

            "stable_score": score.stable_score,

            "aggressive_score": score.aggressive_score,

            "stable_rank": score.stable_rank,

            "aggressive_rank": score.aggressive_rank,

        }



    @staticmethod

    def _bucket_summary(trades: list[dict]) -> dict:

        profit = sum(safe_float(item.get("profit")) or 0.0 for item in trades)

        cost = sum(safe_float(item.get("buy_cost")) or 0.0 for item in trades)

        wins = sum(1 for item in trades if (safe_float(item.get("profit")) or 0.0) > 0)

        return {

            "trades": len(trades),

            "profit": round(profit, 4),

            "return_on_cost": round(profit / cost, 6) if cost else None,

            "win_rate": round(wins / len(trades), 6) if trades else None,

        }



    @staticmethod
    def build_execution_package(
        factor_pool_version: str,
        training_report_path: str | Path = "reports/ml_lab/models_versioned_v2/model_comparison_report.json",
        convergence_report_path: str | Path = "reports/ml_lab/convergence_versioned_v2/factor_convergence_details.json",
        registry_path: str | Path = "reports/ml_lab/versioned_registry.json",
        output_path: str | Path = "reports/ml_lab/execution_package_v1.json",
    ) -> dict:
        training_report = json.loads(Path(training_report_path).read_text(encoding="utf-8"))
        convergence_report = json.loads(Path(convergence_report_path).read_text(encoding="utf-8"))
        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        models = training_report.get("models", [])
        stable_model = next((m for m in models if m.get("model") == "lightgbm" and m.get("target") == "stable"), None)
        aggressive_model = next((m for m in models if m.get("model") == "lightgbm" and m.get("target") == "aggressive"), None)
        summary_rows = convergence_report.get("summary", [])
        package = {
            "factor_pool_version": factor_pool_version,
            "training_frame_path": training_report.get("training_frame_path"),
            "validation_windows": [item.get("window") for item in (stable_model or {}).get("evaluations", [])],
            "stable_model": {
                "model": stable_model.get("model") if stable_model else None,
                "target": "stable",
                "mean_top10_label_rate": round(sum(item.get("top10", {}).get("label_rate", 0.0) for item in (stable_model or {}).get("evaluations", [])) / len((stable_model or {}).get("evaluations", [])), 6) if stable_model else None,
                "mean_top50_return": round(sum(item.get("top50", {}).get("mean_net_return", 0.0) for item in (stable_model or {}).get("evaluations", [])) / len((stable_model or {}).get("evaluations", [])), 6) if stable_model else None,
                "control_top50_return": next((item.get("top50", {}).get("mean_net_return") for item in (stable_model or {}).get("evaluations", []) if item.get("window") == "control_choppy"), None),
            },
            "aggressive_model": {
                "model": aggressive_model.get("model") if aggressive_model else None,
                "target": "aggressive",
                "mean_top10_label_rate": round(sum(item.get("top10", {}).get("label_rate", 0.0) for item in (aggressive_model or {}).get("evaluations", [])) / len((aggressive_model or {}).get("evaluations", [])), 6) if aggressive_model else None,
                "mean_top50_return": round(sum(item.get("top50", {}).get("mean_net_return", 0.0) for item in (aggressive_model or {}).get("evaluations", [])) / len((aggressive_model or {}).get("evaluations", [])), 6) if aggressive_model else None,
                "control_top50_return": next((item.get("top50", {}).get("mean_net_return") for item in (aggressive_model or {}).get("evaluations", []) if item.get("window") == "control_choppy"), None),
            },
            "factor_convergence": {
                "keep": [row for row in summary_rows if row.get("verdict") == "保留"],
                "watch": [row for row in summary_rows if row.get("verdict") == "观察"],
                "downgrade": [row for row in summary_rows if row.get("verdict") == "降级"],
            },
            "registry": registry,
        }
        Path(output_path).write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        return package

    @staticmethod
    def _model_validation_summary(models: list[dict]) -> dict:
        summary = {"trained": 0, "missing_dependency": 0, "single_class_training_target": 0, "windows": []}
        for model in models:
            status = model.get("status")
            summary[status] = summary.get(status, 0) + 1
            for evaluation in model.get("evaluations", []):
                summary["windows"].append({
                    "model": model.get("model"),
                    "target": model.get("target"),
                    "window": evaluation.get("window"),
                    "top10_label_rate": (evaluation.get("top10") or {}).get("label_rate"),
                    "top50_mean_net_return": (evaluation.get("top50") or {}).get("mean_net_return"),
                    "top100_mean_net_return": (evaluation.get("top100") or {}).get("mean_net_return"),
                })
        return summary

    def _write_experiment_ledger(self, output_path: Path, report_path: Path, summary_path: Path, detail_path: Path) -> Path:
        ledger_path = output_path / "experiment_ledger.json"
        payload = {
            "model_version": self.config.model_version,
            "factor_pool_version": self.active_factor_set,
            "training_report": str(report_path),
            "validation_summary": str(summary_path),
            "convergence_report": str(detail_path),
        }
        ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return ledger_path

    def write_versioned_registry(
        self,
        output_path: str | Path,
        factor_pool_version: str,
        factor_pool_source: str = "factor_lab.validation_suite",
        factor_pool_run_id: int | None = None,
        ml_training_report: str | Path = "reports/ml_lab/models_versioned/model_comparison_report.json",
        ml_validation_summary: str | Path = "reports/ml_lab/models_versioned/model_validation_summary.json",
        ml_experiment_ledger: str | Path = "reports/ml_lab/models_versioned/experiment_ledger.json",
        factor_convergence_report: str | Path = "reports/ml_lab/convergence_versioned_v2/factor_convergence_details.json",
        factor_convergence_ledger: str | Path = "reports/ml_lab/convergence_versioned_v2/experiment_ledger.json",
        factor_pool_registry: str | Path = "reports/factor_lab_versioned/candidate_pool_registry.json",
        factor_validation_summary: str | Path = "reports/factor_lab_versioned/factor_validation_summary.csv",
        factor_validation_details: str | Path = "reports/factor_lab_versioned/factor_validation_details.json",
    ) -> dict:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "factor_pool_version": factor_pool_version,
            "factor_pool_source": factor_pool_source,
            "factor_pool_run_id": factor_pool_run_id,
            "ml_model_version": self.config.model_version,
            "ml_training_report": str(ml_training_report),
            "ml_validation_summary": str(ml_validation_summary),
            "ml_experiment_ledger": str(ml_experiment_ledger),
            "factor_convergence_report": str(factor_convergence_report),
            "factor_convergence_ledger": str(factor_convergence_ledger),
            "factor_pool_registry": str(factor_pool_registry),
            "factor_validation_summary": str(factor_validation_summary),
            "factor_validation_details": str(factor_validation_details),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _train_one_model(

        self,

        model_name: str,

        target_name: str,

        label_col: str,

        feature_cols: list[str],

        train_df: pd.DataFrame,

        all_df: pd.DataFrame,

        output_path: Path,

    ) -> dict:

        if not importlib.util.find_spec(model_name):

            return {"model": model_name, "target": target_name, "status": "missing_dependency"}

        target = pd.to_numeric(train_df[label_col], errors="coerce")

        train = train_df[target.notna()].copy()

        target = target[target.notna()].astype(int)

        if target.nunique() < 2:

            return {"model": model_name, "target": target_name, "status": "single_class_training_target"}



        model = self._make_classifier(model_name)

        model.fit(train[feature_cols], target)

        model_file = output_path / f"{model_name}_{target_name}.joblib"

        import joblib



        joblib.dump({"model": model, "features": feature_cols, "label": label_col}, model_file)

        evaluations = []

        for window in VALIDATION_WINDOWS:

            window_df = all_df[

                (all_df["trade_date"].astype(str) >= window["start_date"])

                & (all_df["trade_date"].astype(str) <= window["end_date"])

                & pd.to_numeric(all_df[label_col], errors="coerce").notna()

            ].copy()

            if window_df.empty:

                evaluations.append({"window": window["name"], "rows": 0})

                continue

            window_df[feature_cols] = window_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            proba = self._predict_positive(model, window_df[feature_cols])

            window_df["prediction"] = proba

            evaluations.append(self._evaluate_predictions(window["name"], window_df, label_col))

        return {

            "model": model_name,

            "target": target_name,

            "status": "trained",

            "model_path": str(model_file),

            "train_rows": int(len(train)),

            "positive_rate": round(float(target.mean()), 6),

            "evaluations": evaluations,

        }



    @staticmethod

    def _make_classifier(model_name: str):

        if model_name == "lightgbm":

            from lightgbm import LGBMClassifier



            return LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=5, num_leaves=31, random_state=42, verbose=-1)

        if model_name == "xgboost":

            from xgboost import XGBClassifier



            return XGBClassifier(

                n_estimators=120,

                learning_rate=0.05,

                max_depth=4,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=42,

                eval_metric="logloss",

            )

        if model_name == "catboost":

            from catboost import CatBoostClassifier



            return CatBoostClassifier(iterations=120, learning_rate=0.05, depth=5, random_seed=42, verbose=False)

        raise ValueError(model_name)



    @staticmethod

    def _predict_positive(model, features: pd.DataFrame) -> list[float]:

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(features)

            return [float(row[-1]) for row in proba]

        return [float(value) for value in model.predict(features)]



    @staticmethod

    def _evaluate_predictions(window_name: str, df: pd.DataFrame, label_col: str) -> dict:

        ranked = df.sort_values("prediction", ascending=False)

        result = {"window": window_name, "rows": int(len(ranked))}

        for top_n in [10, 50, 100]:

            top = ranked.head(top_n)

            result[f"top{top_n}"] = {

                "rows": int(len(top)),

                "label_rate": round(float(pd.to_numeric(top[label_col], errors="coerce").mean()), 6) if not top.empty else None,

                "mean_net_return": round(float(pd.to_numeric(top["label_net_return"], errors="coerce").mean()), 6) if not top.empty else None,

            }

        return result



    @staticmethod

    def _is_validation_date(trade_date: str) -> bool:

        return any(window["start_date"] <= trade_date <= window["end_date"] for window in VALIDATION_WINDOWS)



    @staticmethod

    def feature_names(factor_set: str = "factor_lab_v1") -> list[str]:

        return FactorLabService.factor_names_for_set(factor_set)

    @staticmethod
    def build_feature_manifest(feature_names: list[str]) -> dict:
        groups = {
            "moneyflow": {"main_net", "large_order", "extra_large", "moneyflow"},
            "industry": {"relative_industry", "industry_rank", "cross_industry"},
            "index_daily": {"market_return", "market_volatility", "market_drawdown"},
            "fina_indicator": {"roe", "roa", "roic", "grossprofit", "netprofit", "ocf", "debt", "current", "quick", "tr_yoy", "or_yoy", "assets", "eqt", "rd_exp", "q_"},
            "daily_basic": {"amount", "turnover", "volume_ratio", "circ_mv", "pe", "pb", "ps", "dv", "free_share"},
            "quote_history": {"return", "reversal", "ma_gap", "ma_slope", "distance", "drawdown", "volatility", "downside", "amplitude", "amount_avg", "amount_zscore", "amihud", "limit", "close_position"},
            "derived": {"quality", "growth", "value", "cross_quality", "cross_return"},
        }
        rows = []
        for name in feature_names:
            source = "derived"
            for candidate, hints in groups.items():
                if any(name.startswith(hint) or hint in name for hint in hints):
                    source = candidate
                    break
            rows.append(
                {
                    "name": name,
                    "column": f"feature_{name}",
                    "source": source,
                    "required": not MlLabService.is_optional_feature(name),
                    "availability_policy": "signal_date_or_prior_only; financial features require ann_date <= trade_date",
                    "leakage_eligible": False,
                }
            )
        return {
            "feature_count": len(feature_names),
            "features": rows,
            "missingness_policy": {
                "required_max_missing_rate": 0.05,
                "optional_max_missing_rate": 0.20,
                "action": "drop or block features that fail the contract; never silently train on 100%-missing features",
            },
        }

    @staticmethod
    def is_optional_feature(name_or_column: str) -> bool:
        return is_optional_feature(name_or_column)

    @staticmethod
    def feature_missingness_gate(
        frame: pd.DataFrame,
        feature_cols: list[str] | None = None,
        required_max_missing_rate: float = 0.05,
        optional_max_missing_rate: float = 0.20,
    ) -> dict:
        cols = feature_cols or sorted([col for col in frame.columns if col.startswith("feature_")])
        rows = []
        blocking = []
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(dtype=float)
            missing_rate = float(values.isna().mean()) if len(values) else 1.0
            non_null_rows = int(values.notna().sum())
            optional = MlLabService.is_optional_feature(col)
            threshold = optional_max_missing_rate if optional else required_max_missing_rate
            item = {
                "column": col,
                "missing_rate": round(missing_rate, 6),
                "non_null_rows": non_null_rows,
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
            "optional_max_missing_rate": optional_max_missing_rate,
            "blocking_count": len(blocking),
            "blocking_features": blocking,
            "feature_missingness": rows,
            "policy": "Required active training features must not exceed the missingness threshold; 100%-missing features are blocked.",
        }



    @staticmethod

    def _trade_dates(start_date: str, end_date: str, limit: int | None = None) -> list[str]:

        with session_scope() as session:

            query = (

                session.query(DailyQuote.trade_date)

                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)

                .group_by(DailyQuote.trade_date)

                .order_by(DailyQuote.trade_date.asc())

            )

            if limit:

                query = query.limit(limit)

            rows = query.all()

        return [row[0] for row in rows]



    @staticmethod

    def model_dependencies() -> dict[str, bool]:

        return {

            name: importlib.util.find_spec(name) is not None

            for name in ["lightgbm", "catboost", "xgboost", "sklearn", "torch"]

        }

