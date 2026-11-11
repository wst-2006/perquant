"""Reproducible audit of historical feature availability and model membership.

This module reports what is actually stored locally.  It deliberately keeps
missing values visible and uses ``ann_date < signal_date`` for financial and
announcement data because a date-only source cannot prove intraday timing.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func

from ..data.tushare_client import FINA_INDICATOR_FACTOR_NAMES
from ..db import DailyQuote, Stock, TushareRawRecord, session_scope
from ..factor_lab.metrics import safe_float
from ..factor_lab.service import _select_latest_financial_payloads
from .v8_events import V8_EVENT_ENDPOINTS, V8_EVENT_FEATURES
from .v9_events import V9_ANNOUNCEMENT_ENDPOINTS, V9_EVENT_ENDPOINTS, V9_EVENT_FEATURES, V9_TRADE_DATE_ENDPOINTS


AUDIT_CONTRACT = "ml_data_availability_audit_v1"
DEFAULT_CHECKPOINTS = ("20240401", "20250603", "20260601")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_range(rows: list[tuple[str | None, str | None, str | None]]) -> dict[str, str | None]:
    values = [str(value) for row in rows for value in row if value and len(str(value)) == 8]
    return {"start": min(values) if values else None, "end": max(values) if values else None}


def _availability_status(coverage: float | None, *, sufficient_at: float = 0.90) -> str:
    if coverage is None:
        return "not_derivable"
    if coverage >= sufficient_at:
        return "historically_sufficient"
    return "historically_insufficient"


def _model_membership(feature_columns: set[str], candidates: tuple[str, ...]) -> dict[str, Any]:
    used = [name for name in candidates if name in feature_columns]
    return {"currently_used_by_active_model": bool(used), "used_feature_count": len(used), "used_features": used}


class DataAvailabilityAuditor:
    def __init__(
        self,
        *,
        artifact_path: str | Path = "reports/ml_lab/v11_dual_artifact_feature_v2/artifact.json",
        v8_manifest_path: str | Path = "reports/ml_lab/training_frame_v8_events/manifest.json",
        v9_manifest_path: str | Path = "reports/ml_lab/training_frame_v9_events/manifest.json",
        checkpoints: tuple[str, ...] = DEFAULT_CHECKPOINTS,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.v8_manifest_path = Path(v8_manifest_path)
        self.v9_manifest_path = Path(v9_manifest_path)
        self.checkpoints = tuple(sorted({str(value) for value in checkpoints if str(value).isdigit() and len(str(value)) == 8}))

    def run(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact = _read_json(self.artifact_path) or {}
        v8_manifest = _read_json(self.v8_manifest_path) or {}
        v9_manifest = _read_json(self.v9_manifest_path) or {}
        feature_columns = set(artifact.get("feature_columns") or [])
        report = {
            "audit_contract": AUDIT_CONTRACT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "audit_scope": {
                "financial_and_announcement_policy": "ann_date strictly earlier than signal_date; same-day publication is excluded because local raw records do not include a disclosure timestamp.",
                "financial_sufficiency_threshold": 0.90,
                "checkpoint_dates": list(self.checkpoints),
                "checkpoint_note": "These are reproducible checkpoints, not a claim of full daily historical coverage.",
            },
            "active_model": self._active_model(artifact, feature_columns),
            "financial": self._financial_audit(feature_columns, output),
            "events": self._event_audit(feature_columns, v8_manifest, v9_manifest),
            "training_datasets": self._dataset_membership(v8_manifest, v9_manifest),
            "conclusion": {},
        }
        report["conclusion"] = self._conclusion(report)
        json_path = output / "data_availability_audit.json"
        markdown_path = output / "data_availability_audit.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(self._markdown(report), encoding="utf-8")
        return {**report, "json_path": str(json_path), "markdown_path": str(markdown_path)}

    def _active_model(self, artifact: dict[str, Any], feature_columns: set[str]) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": _file_sha256(self.artifact_path),
            "model_version": artifact.get("model_version"),
            "feature_set": artifact.get("feature_set"),
            "feature_count": len(feature_columns),
            "diagnostic_only": bool(artifact.get("diagnostic_only", True)),
            "promotion_allowed": bool(artifact.get("promotion_allowed", False)),
        }

    def _financial_audit(self, feature_columns: set[str], output_dir: Path) -> dict[str, Any]:
        checkpoints = []
        with session_scope() as session:
            raw_rows = int(
                session.query(func.count(TushareRawRecord.id))
                .filter(TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip")))
                .scalar()
                or 0
            )
            raw_codes = int(
                session.query(func.count(func.distinct(TushareRawRecord.ts_code)))
                .filter(TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip")), TushareRawRecord.ts_code.isnot(None))
                .scalar()
                or 0
            )
            announcement_date_rows = session.query(
                TushareRawRecord.ann_date, TushareRawRecord.f_ann_date
            ).filter(TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip"))).all()
            period_date_rows = session.query(TushareRawRecord.end_date).filter(
                TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip"))
            ).all()
            for signal_date in self.checkpoints:
                universe = {
                    row[0]
                    for row in session.query(DailyQuote.ts_code).filter(DailyQuote.trade_date == signal_date).all()
                }
                if not universe:
                    checkpoints.append(
                        {
                            "signal_date": signal_date,
                            "checkpoint_status": "not_a_local_trade_date",
                            "quoted_universe_count": 0,
                            "latest_eligible_financial_stock_count": 0,
                            "latest_eligible_financial_coverage": None,
                            "fields": [],
                        }
                    )
                    continue
                eligible_rows = (
                    session.query(TushareRawRecord)
                    .filter(
                        TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip")),
                        TushareRawRecord.ann_date < signal_date,
                    )
                    .all()
                )
                latest = _select_latest_financial_payloads(eligible_rows)
                latest = {code: payload for code, payload in latest.items() if code in universe}
                fields = []
                for field in FINA_INDICATOR_FACTOR_NAMES:
                    count = sum(safe_float(payload.get(field)) is not None for payload in latest.values())
                    coverage = count / len(universe) if universe else None
                    fields.append(
                        {
                            "field": field,
                            "covered_stock_count": count,
                            "universe_stock_count": len(universe),
                            "coverage": coverage,
                            "status": _availability_status(coverage),
                        }
                    )
                checkpoints.append(
                    {
                        "signal_date": signal_date,
                        "quoted_universe_count": len(universe),
                        "latest_eligible_financial_stock_count": len(latest),
                        "latest_eligible_financial_coverage": len(latest) / len(universe) if universe else None,
                        "fields": fields,
                    }
                )
        membership = _model_membership(feature_columns, tuple(f"feature_{name}" for name in FINA_INDICATOR_FACTOR_NAMES))
        full_history = self._full_history_financial_coverage(output_dir)
        return {
            "raw_available": raw_rows > 0,
            "raw_row_count": raw_rows,
            "raw_stock_count": raw_codes,
            "raw_announcement_date_range": _date_range(announcement_date_rows),
            "raw_reporting_period_range": _date_range(period_date_rows),
            "point_in_time_derivable": raw_rows > 0,
            "selection_policy": "latest eligible report selected by end_date, ann_date, field richness, update timestamp and VIP preference; eligibility requires ann_date < signal_date.",
            "revision_timing_audit": {
                "status": "blocked",
                "reason": "Local updated_at is the synchronization timestamp, not a verified official revision disclosure timestamp. The current raw financial contract does not prove that later revisions were known at every historical signal date.",
                "impact": "Financial field coverage alone is insufficient to add these fields to historical model training. A revision-safe source/version contract is required first.",
            },
            "current_factor_lab_policy": "strictly earlier than trade_date after this audit change; legacy financial fallback follows the same policy.",
            "active_model_membership": membership,
            "checkpoints": checkpoints,
            "full_training_history": full_history,
        }

    def _training_dates(self) -> list[str]:
        parent = self.v9_manifest_path.parent
        dates = []
        for path in parent.glob("trade_date=*"):
            value = path.name.split("=", 1)[-1]
            if len(value) == 8 and value.isdigit():
                dates.append(value)
        return sorted(set(dates))

    @staticmethod
    def _financial_priority(row: TushareRawRecord) -> tuple[Any, ...]:
        payload = dict(row.payload or {})
        end_date = str(row.end_date or payload.get("end_date") or "")
        ann_date = str(row.ann_date or payload.get("ann_date") or "")
        richness = sum(safe_float(payload.get(name)) is not None for name in FINA_INDICATOR_FACTOR_NAMES)
        endpoint_priority = 1 if row.endpoint == "fina_indicator_vip" else 0
        return end_date, ann_date, richness, str(row.updated_at or ""), endpoint_priority

    def _full_history_financial_coverage(self, output_dir: Path) -> dict[str, Any]:
        """Audit every frozen training date without making a model-side assumption."""
        signal_dates = self._training_dates()
        if not signal_dates:
            return {"available": False, "reason": "No trade_date partitions found beside v9 manifest."}
        start_date, end_date = signal_dates[0], signal_dates[-1]
        with session_scope() as session:
            raw_rows = (
                session.query(TushareRawRecord)
                .filter(
                    TushareRawRecord.endpoint.in_(("fina_indicator", "fina_indicator_vip")),
                    TushareRawRecord.ann_date.isnot(None),
                    TushareRawRecord.ann_date < end_date,
                )
                .all()
            )
            quote_rows = (
                session.query(DailyQuote.trade_date, DailyQuote.ts_code)
                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)
                .order_by(DailyQuote.trade_date.asc())
                .all()
            )
        universes: dict[str, set[str]] = {date: set() for date in signal_dates}
        for trade_date, ts_code in quote_rows:
            if trade_date in universes:
                universes[trade_date].add(ts_code)
        ordered_rows = sorted(
            (row for row in raw_rows if row.ann_date and len(str(row.ann_date)) == 8),
            key=lambda row: (str(row.ann_date), self._financial_priority(row), str(row.ts_code or "")),
        )
        cursor = 0
        selected: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        timeline: list[dict[str, Any]] = []
        for signal_date in signal_dates:
            while cursor < len(ordered_rows) and str(ordered_rows[cursor].ann_date) < signal_date:
                row = ordered_rows[cursor]
                cursor += 1
                code = str(row.ts_code or "")
                if not code:
                    continue
                priority = self._financial_priority(row)
                current = selected.get(code)
                if current is None or priority > current[0]:
                    selected[code] = (priority, dict(row.payload or {}))
            universe = universes[signal_date]
            selected_payloads = [payload for code, (_, payload) in selected.items() if code in universe]
            field_coverage = {
                field: (sum(safe_float(payload.get(field)) is not None for payload in selected_payloads) / len(universe) if universe else None)
                for field in FINA_INDICATOR_FACTOR_NAMES
            }
            timeline.append(
                {
                    "signal_date": signal_date,
                    "universe_stock_count": len(universe),
                    "latest_eligible_financial_stock_count": len(selected_payloads),
                    "latest_eligible_financial_coverage": len(selected_payloads) / len(universe) if universe else None,
                    "field_coverage": field_coverage,
                }
            )
        csv_path = output_dir / "financial_coverage_timeline.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "signal_date",
                    "universe_stock_count",
                    "latest_eligible_financial_stock_count",
                    "latest_eligible_financial_coverage",
                    *FINA_INDICATOR_FACTOR_NAMES,
                ],
            )
            writer.writeheader()
            for item in timeline:
                writer.writerow(
                    {
                        "signal_date": item["signal_date"],
                        "universe_stock_count": item["universe_stock_count"],
                        "latest_eligible_financial_stock_count": item["latest_eligible_financial_stock_count"],
                        "latest_eligible_financial_coverage": item["latest_eligible_financial_coverage"],
                        **item["field_coverage"],
                    }
                )
        field_summary = []
        for field in FINA_INDICATOR_FACTOR_NAMES:
            values = [item["field_coverage"][field] for item in timeline if item["field_coverage"][field] is not None]
            below_dates = [item["signal_date"] for item in timeline if (item["field_coverage"][field] or 0) < 0.90]
            field_summary.append(
                {
                    "field": field,
                    "min_coverage": min(values) if values else None,
                    "max_coverage": max(values) if values else None,
                    "days_below_90pct": len(below_dates),
                    "first_below_90pct": min(below_dates) if below_dates else None,
                    "last_below_90pct": max(below_dates) if below_dates else None,
                    "status": _availability_status(min(values) if values else None),
                }
            )
        return {
            "available": True,
            "date_count": len(timeline),
            "date_range": {"start": start_date, "end": end_date},
            "timeline_csv_path": str(csv_path),
            "timeline_csv_sha256": _file_sha256(csv_path),
            "field_summary": field_summary,
            "policy": "All frozen v9 training dates; every date uses only financial rows with ann_date strictly earlier than signal_date and its own quoted A-share universe.",
        }

    def _event_audit(self, feature_columns: set[str], v8_manifest: dict[str, Any], v9_manifest: dict[str, Any]) -> dict[str, Any]:
        endpoint_rows: dict[str, dict[str, Any]] = {}
        all_endpoints = tuple((*V8_EVENT_ENDPOINTS, *V9_EVENT_ENDPOINTS))
        with session_scope() as session:
            for endpoint in all_endpoints:
                rows = (
                    session.query(TushareRawRecord.trade_date, TushareRawRecord.ann_date, TushareRawRecord.f_ann_date, TushareRawRecord.end_date)
                    .filter(TushareRawRecord.endpoint == endpoint)
                    .all()
                )
                count = len(rows)
                code_count = int(
                    session.query(func.count(func.distinct(TushareRawRecord.ts_code)))
                    .filter(TushareRawRecord.endpoint == endpoint, TushareRawRecord.ts_code.isnot(None))
                    .scalar()
                    or 0
                )
                if endpoint in V9_TRADE_DATE_ENDPOINTS:
                    available_dates = [(row[0],) for row in rows]
                else:
                    available_dates = [(row[1], row[2]) for row in rows]
                endpoint_rows[endpoint] = {
                    "raw_available": count > 0,
                    "raw_row_count": count,
                    "raw_stock_count": code_count,
                    "availability_date_range": _date_range(available_dates),
                    "reporting_or_event_date_range": _date_range([(row[3],) for row in rows]),
                    "availability_type": "trade_date_after_close" if endpoint in V9_TRADE_DATE_ENDPOINTS else "announcement_strictly_before_next_signal",
                }
        v8_membership = _model_membership(feature_columns, V8_EVENT_FEATURES)
        v9_membership = _model_membership(feature_columns, V9_EVENT_FEATURES)
        return {
            "strict_announcement_policy": "Announcement features require ann_date < signal_date. Trade-date endpoints are only used after the market close of their trade_date.",
            "endpoints": endpoint_rows,
            "v8_announcement_features": {
                "dataset_manifest": self._dataset_summary(v8_manifest, self.v8_manifest_path),
                "active_model_membership": v8_membership,
                "feature_count": len(V8_EVENT_FEATURES),
            },
            "v9_structural_features": {
                "dataset_manifest": self._dataset_summary(v9_manifest, self.v9_manifest_path),
                "active_model_membership": v9_membership,
                "feature_count": len(V9_EVENT_FEATURES),
                "coverage_warning": "Discrete events are not expected every day. Raw row presence does not establish uniform issuer or historical coverage; inspect endpoint date ranges and frozen-manifest non-null counts before any merge.",
            },
            "announcement_endpoints": list(V9_ANNOUNCEMENT_ENDPOINTS),
        }

    def _dataset_membership(self, v8_manifest: dict[str, Any], v9_manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "v8_events": self._dataset_summary(v8_manifest, self.v8_manifest_path),
            "v9_events": self._dataset_summary(v9_manifest, self.v9_manifest_path),
            "merge_status": "not_verified",
            "merge_requirement": "Do not merge v8 and v9 from matching row counts alone. Verify exact (trade_date, ts_code) key equality and label/base-feature hashes first.",
        }

    @staticmethod
    def _dataset_summary(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
        if not manifest:
            return {"available": False, "manifest_path": str(path)}
        return {
            "available": True,
            "manifest_path": str(path),
            "manifest_sha256": _file_sha256(path),
            "feature_set": manifest.get("feature_set"),
            "rows": manifest.get("rows"),
            "date_count": manifest.get("date_count"),
            "date_range": manifest.get("date_range"),
            "event_endpoints": manifest.get("event_endpoints"),
            "event_feature_non_null_counts": manifest.get("event_feature_non_null_counts"),
            "announcement_policy": manifest.get("announcement_policy") or manifest.get("availability_policy"),
        }

    @staticmethod
    def _conclusion(report: dict[str, Any]) -> dict[str, Any]:
        financial = report["financial"]
        checkpoints = financial["checkpoints"]
        history = financial.get("full_training_history") or {}
        field_summary = history.get("field_summary") or []
        insufficient = sorted(field["field"] for field in field_summary if field["status"] != "historically_sufficient")
        return {
            "status": "audit_complete",
            "historical_financial_layer": "原始财务数据真实存在且可按公告日严格推导，但并非所有字段都达到完整历史覆盖阈值。",
            "historical_financial_revision_timing": "blocked",
            "current_v11_uses_financial_features": financial["active_model_membership"]["currently_used_by_active_model"],
            "current_v11_uses_v8_announcement_features": report["events"]["v8_announcement_features"]["active_model_membership"]["currently_used_by_active_model"],
            "current_v11_uses_v9_structural_features": report["events"]["v9_structural_features"]["active_model_membership"]["currently_used_by_active_model"],
            "financial_fields_below_90pct_in_full_training_history": insufficient,
            "full_training_history_dates_audited": history.get("date_count", 0),
            "next_action": "先补齐官方财务修订披露时点的可审计合同，再冻结财务侧车字段与覆盖阈值；不得用近期快照回填历史，也不得在此之前把财务字段加入模型调参。",
        }

    @staticmethod
    def _markdown(report: dict[str, Any]) -> str:
        conclusion = report["conclusion"]
        lines = [
            "# 历史数据可用性审计",
            "",
            "## 结论",
            "",
            f"- 历史财务层：{conclusion['historical_financial_layer']}",
            f"- 财务修订时间审计：{conclusion['historical_financial_revision_timing']}",
            f"- 当前 v11 使用财务因子：{conclusion['current_v11_uses_financial_features']}",
            f"- 当前 v11 使用 v8 公告因子：{conclusion['current_v11_uses_v8_announcement_features']}",
            f"- 当前 v11 使用 v9 结构事件：{conclusion['current_v11_uses_v9_structural_features']}",
            f"- 全训练期审计交易日：{conclusion['full_training_history_dates_audited']}",
            f"- 下一步：{conclusion['next_action']}",
            "",
            "## 时间可得性规则",
            "",
            report["audit_scope"]["financial_and_announcement_policy"],
            "",
            "## 财务检查点",
            "",
            "| 信号日 | 行情股票数 | 有可用财务报表数 | 覆盖率 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for item in report["financial"]["checkpoints"]:
            coverage = item["latest_eligible_financial_coverage"]
            lines.append(f"| {item['signal_date']} | {item['quoted_universe_count']} | {item['latest_eligible_financial_stock_count']} | {coverage:.2%} |" if coverage is not None else f"| {item['signal_date']} | 0 | 0 | -- |")
        lines.extend(["", "字段覆盖详见同目录 JSON；检查点不是完整日度覆盖的替代。", "", "## 事件端点", ""])
        for endpoint, item in report["events"]["endpoints"].items():
            date_range = item["availability_date_range"]
            lines.append(f"- `{endpoint}`：{item['raw_row_count']} 行，{item['raw_stock_count']} 只，可用日期 {date_range['start']} 至 {date_range['end']}，{item['availability_type']}。")
        return "\n".join(lines) + "\n"
