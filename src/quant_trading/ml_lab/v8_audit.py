"""Audits for the partitioned v8 dataset and its path labels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..db import Stock, session_scope
from ..services.data_sync import _asset_class_for_ts_code
from .v8_labels import (
    V8CostConfig,
    V8_HORIZONS,
    V8_LABEL_CONTRACT,
    V8_THRESHOLDS,
    V8MarketPanel,
    compute_path_labels,
)


V8_AUDIT_CONTRACT = "ml_v8_dataset_audit_v1"


class V8DatasetAuditError(RuntimeError):
    """Raised when the v8 dataset cannot be audited."""


class V8DatasetAuditor:
    def run(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        *,
        sample_size: int = 10_000,
        random_state: int = 42,
        tolerance: float = 5e-6,
    ) -> dict[str, Any]:
        root = Path(dataset_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise V8DatasetAuditError(f"Missing v8 manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("label_contract") != V8_LABEL_CONTRACT:
            raise V8DatasetAuditError(f"Unsupported label contract: {manifest.get('label_contract')}")

        partitions = sorted(root.glob("trade_date=*/part-000.parquet"), key=lambda path: path.parent.name)
        if not partitions:
            raise V8DatasetAuditError(f"No v8 Parquet partitions found in {root}")
        scan = self._scan_partitions(partitions, manifest, sample_size, random_state)
        universe = self._audit_universe(scan.pop("codes"))
        sample = scan.pop("sample")
        reconstruction = self._reconstruct_sample(sample, manifest, tolerance)
        issues = list(scan.pop("issues"))
        if universe["status"] != "pass":
            issues.extend(universe["issues"])
        if reconstruction["status"] != "pass":
            issues.append("label_reconstruction_mismatch")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        partition_manifest_path = output / "partition_manifest.json"
        partition_manifest_path.write_text(
            json.dumps(scan.pop("partition_manifest"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload = {
            "task": "v8_dataset_audit",
            "audit_contract": V8_AUDIT_CONTRACT,
            "status": "pass" if not issues else "blocked",
            "issues": sorted(set(issues)),
            "dataset_dir": str(root),
            "dataset_manifest_path": str(manifest_path),
            "source_sha256": manifest.get("source_sha256"),
            "dataset_sha256": scan.pop("dataset_sha256"),
            "schema_sha256": scan.pop("schema_sha256"),
            "partition_manifest_path": str(partition_manifest_path),
            "partition_manifest_sha256": _file_sha256(partition_manifest_path),
            "scan": scan,
            "universe_audit": universe,
            "label_reconstruction": reconstruction,
            "diagnostic_only": True,
            "promotion_allowed": False,
            "policy": "Real Tushare-backed A-share rows only; dataset research is blocked unless every audit passes.",
        }
        report_path = output / "v8_dataset_audit.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {**payload, "report_path": str(report_path)}

    @staticmethod
    def validate_report(report_path: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise V8DatasetAuditError(f"Missing required v8 dataset audit: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("audit_contract") != V8_AUDIT_CONTRACT:
            raise V8DatasetAuditError(f"Unsupported v8 audit contract: {report.get('audit_contract')}")
        expected = Path(dataset_dir).resolve()
        audited = Path(report.get("dataset_dir") or "").resolve()
        if audited != expected:
            raise V8DatasetAuditError(f"Audit dataset mismatch: audited={audited}, requested={expected}")
        if report.get("status") != "pass":
            raise V8DatasetAuditError(f"v8 dataset audit is blocked: {report.get('issues')}")
        return report

    def _scan_partitions(
        self,
        partitions: list[Path],
        manifest: dict[str, Any],
        sample_size: int,
        random_state: int,
    ) -> dict[str, Any]:
        canonical_schema = pq.ParquetFile(partitions[0]).schema_arrow
        schema_text = str(canonical_schema)
        schema_hash = sha256(schema_text.encode("utf-8")).hexdigest()
        columns = canonical_schema.names
        feature_columns = sorted(name for name in columns if name.startswith("feature_"))
        label_columns = sorted(name for name in columns if name.startswith("label_"))
        required = {
            "trade_date",
            "ts_code",
            "industry",
            "label_entry_date",
            "label_entry_tradable",
            "label_5d_net_return",
            "label_10d_mfe_close",
            "label_20d_hit_8pct",
        }
        issues: list[str] = []
        if missing := sorted(required - set(columns)):
            issues.append(f"missing_required_columns:{','.join(missing)}")
        non_float32_features = [
            field.name for field in canonical_schema if field.name in feature_columns and str(field.type) != "float"
        ]
        if non_float32_features:
            issues.append("feature_dtype_not_float32")

        row_count = 0
        duplicate_keys = 0
        partition_date_mismatches = 0
        schema_mismatches: list[str] = []
        temporal_issues = Counter()
        feature_non_null = Counter()
        label_non_null = Counter()
        rows_by_date: dict[str, int] = {}
        codes: set[str] = set()
        sample_parts: list[pd.DataFrame] = []
        partition_manifest: list[dict[str, Any]] = []
        dataset_digest = sha256()
        per_partition_sample = max(1, int(np.ceil(sample_size * 1.5 / len(partitions))))
        rng = np.random.default_rng(random_state)

        for path in partitions:
            partition_date = path.parent.name.split("=", 1)[1]
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(canonical_schema):
                schema_mismatches.append(partition_date)
            file_hash = _file_sha256(path)
            relative_path = path.relative_to(path.parents[1]).as_posix()
            dataset_digest.update(f"{relative_path}|{path.stat().st_size}|{file_hash}\n".encode("utf-8"))
            frame = pd.read_parquet(path)
            rows = len(frame)
            row_count += rows
            rows_by_date[partition_date] = rows
            duplicate_keys += int(frame.duplicated(["trade_date", "ts_code"]).sum())
            normalized_dates = frame["trade_date"].astype(str).str.replace(".0", "", regex=False).str[:8]
            partition_date_mismatches += int((normalized_dates != partition_date).sum())
            codes.update(frame["ts_code"].dropna().astype(str))
            feature_non_null.update({name: int(frame[name].notna().sum()) for name in feature_columns})
            label_non_null.update({name: int(frame[name].notna().sum()) for name in label_columns})
            self._update_temporal_checks(frame, temporal_issues)

            take = min(per_partition_sample, rows)
            if take:
                sample_parts.append(frame.iloc[rng.choice(rows, size=take, replace=False)].copy())
            partition_manifest.append(
                {
                    "trade_date": partition_date,
                    "path": str(path),
                    "rows": rows,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_hash,
                }
            )

        if schema_mismatches:
            issues.append("partition_schema_mismatch")
        if duplicate_keys:
            issues.append("duplicate_trade_date_ts_code")
        if partition_date_mismatches:
            issues.append("partition_date_mismatch")
        if int(manifest.get("rows") or -1) != row_count:
            issues.append("manifest_row_count_mismatch")
        if int(manifest.get("date_count") or -1) != len(partitions):
            issues.append("manifest_partition_count_mismatch")
        empty_features = sorted(name for name in feature_columns if feature_non_null[name] == 0)
        if empty_features:
            issues.append("entirely_missing_features")
        if temporal_issues:
            issues.extend(f"{name}:{count}" for name, count in sorted(temporal_issues.items()) if count)

        sample = pd.concat(sample_parts, ignore_index=True)
        if len(sample) > sample_size:
            sample = sample.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        return {
            "issues": issues,
            "codes": codes,
            "sample": sample,
            "dataset_sha256": dataset_digest.hexdigest(),
            "schema_sha256": schema_hash,
            "partition_manifest": {
                "contract": "ml_v8_partition_manifest_v1",
                "dataset_sha256": dataset_digest.hexdigest(),
                "schema_sha256": schema_hash,
                "partitions": partition_manifest,
            },
            "rows": row_count,
            "date_count": len(partitions),
            "date_range": {"start": min(rows_by_date), "end": max(rows_by_date)},
            "rows_per_date": {
                "min": min(rows_by_date.values()),
                "median": int(np.median(list(rows_by_date.values()))),
                "max": max(rows_by_date.values()),
            },
            "column_count": len(columns),
            "feature_count": len(feature_columns),
            "label_count": len(label_columns),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "non_float32_features": non_float32_features,
            "entirely_missing_features": empty_features,
            "feature_missingness": {
                name: round(1.0 - feature_non_null[name] / row_count, 8) for name in feature_columns
            },
            "label_non_null_counts": dict(sorted(label_non_null.items())),
            "duplicate_key_rows": duplicate_keys,
            "partition_date_mismatch_rows": partition_date_mismatches,
            "schema_mismatch_partitions": schema_mismatches,
            "temporal_consistency_issues": dict(sorted(temporal_issues.items())),
        }

    @staticmethod
    def _update_temporal_checks(frame: pd.DataFrame, issues: Counter) -> None:
        signal = frame["trade_date"].astype(str).str.replace(".0", "", regex=False).str[:8]
        entry = frame["label_entry_date"].astype("string")
        issues["entry_not_after_signal"] += int((entry.notna() & (entry <= signal)).sum())
        tradable = _bool_series(frame["label_entry_tradable"])
        net_columns = [name for name in frame.columns if name.endswith("_net_return")]
        hit_columns = [name for name in frame.columns if "_hit_" in name]
        if net_columns:
            issues["nontradable_entry_has_return"] += int(frame.loc[~tradable, net_columns].notna().any(axis=1).sum())
        if hit_columns:
            issues["nontradable_entry_has_hit_label"] += int(frame.loc[~tradable, hit_columns].notna().any(axis=1).sum())
        for horizon in V8_HORIZONS:
            prefix = f"label_{horizon}d"
            end = frame[f"{prefix}_end_date"].astype("string")
            issues[f"{horizon}d_end_not_after_entry"] += int((end.notna() & entry.notna() & (end <= entry)).sum())
            fixed = _bool_series(frame[f"{prefix}_fixed_tradable"])
            returns = pd.to_numeric(frame[f"{prefix}_net_return"], errors="coerce")
            issues[f"{horizon}d_untradable_exit_has_return"] += int((~fixed & returns.notna()).sum())
            for threshold in V8_THRESHOLDS:
                key = f"{int(round(threshold * 100))}pct"
                hit = frame[f"{prefix}_hit_{key}"]
                time_to = pd.to_numeric(frame[f"{prefix}_time_to_{key}"], errors="coerce")
                hit_bool = _bool_series(hit)
                issues[f"{horizon}d_hit_without_time_{key}"] += int((hit_bool & time_to.isna()).sum())
                issues[f"{horizon}d_invalid_time_{key}"] += int((time_to.notna() & ((time_to < 1) | (time_to > horizon))).sum())

    @staticmethod
    def _audit_universe(codes: set[str], preview_limit: int = 20) -> dict[str, Any]:
        with session_scope() as session:
            rows = session.query(Stock).filter(Stock.ts_code.in_(sorted(codes))).all() if codes else []
        stocks = {row.ts_code: row for row in rows}
        missing = sorted(codes - set(stocks))
        unknown = sorted(
            code for code, row in stocks.items() if row.list_status == "U" or str(row.name or "").startswith("UNKNOWN_")
        )
        quarantined = sorted(code for code, row in stocks.items() if row.list_status == "N")
        unsupported = sorted(code for code, row in stocks.items() if row.list_status not in {"L", "D"})
        non_a_share = sorted(code for code in codes if _asset_class_for_ts_code(code) != "a_share_like")
        issues = []
        for condition, name in (
            (missing, "missing_stock_metadata"),
            (unknown, "unknown_stock_metadata"),
            (quarantined, "quarantined_non_stock_assets"),
            (unsupported, "unsupported_stock_status"),
            (non_a_share, "non_a_share_codes"),
        ):
            if condition:
                issues.append(name)
        return {
            "status": "pass" if not issues else "blocked",
            "issues": issues,
            "unique_codes": len(codes),
            "missing_stock_metadata_count": len(missing),
            "unknown_stock_metadata_count": len(unknown),
            "quarantined_non_stock_asset_count": len(quarantined),
            "unsupported_stock_status_count": len(unsupported),
            "non_a_share_code_count": len(non_a_share),
            "previews": {
                "missing": missing[:preview_limit],
                "unknown": unknown[:preview_limit],
                "quarantined": quarantined[:preview_limit],
                "unsupported": unsupported[:preview_limit],
                "non_a_share": non_a_share[:preview_limit],
            },
        }

    def _reconstruct_sample(
        self,
        sample: pd.DataFrame,
        manifest: dict[str, Any],
        tolerance: float,
    ) -> dict[str, Any]:
        if sample.empty:
            return {"status": "blocked", "sample_rows": 0, "mismatch_count": 0, "mismatches": []}
        start_date = sample["trade_date"].astype(str).min()
        panel = V8MarketPanel.from_database(start_date=start_date)
        costs = V8CostConfig(**(manifest.get("costs") or {}))
        mismatch_counts = Counter()
        mismatches: list[dict[str, Any]] = []
        compared_values = 0
        for row in sample.to_dict("records"):
            expected = self._reference_row(panel, row["ts_code"], row["trade_date"], costs)
            for column, reconstructed in expected.items():
                compared_values += 1
                exported = row.get(column)
                if _values_match(exported, reconstructed, tolerance):
                    continue
                mismatch_counts[column] += 1
                if len(mismatches) < 100:
                    mismatches.append(
                        {
                            "trade_date": str(row["trade_date"]),
                            "ts_code": row["ts_code"],
                            "column": column,
                            "exported": _json_value(exported),
                            "reconstructed": _json_value(reconstructed),
                        }
                    )
        mismatch_count = int(sum(mismatch_counts.values()))
        return {
            "status": "pass" if mismatch_count == 0 else "blocked",
            "sample_rows": int(len(sample)),
            "compared_values": compared_values,
            "tolerance": tolerance,
            "mismatch_count": mismatch_count,
            "mismatch_counts_by_column": dict(sorted(mismatch_counts.items())),
            "mismatches": mismatches,
            "method": "Reference row-by-row path calculation against real PostgreSQL quotes and stock-specific limits.",
        }

    @staticmethod
    def _reference_row(panel: V8MarketPanel, ts_code: str, signal_date: Any, costs: V8CostConfig) -> dict[str, Any]:
        code_index = panel.code_to_index.get(str(ts_code))
        date_index = panel.date_to_index.get(str(signal_date).replace(".0", "")[:8])
        if code_index is None or date_index is None:
            return compute_path_labels(None, [], costs=costs)

        def day(position: int) -> dict[str, Any] | None:
            if position >= len(panel.dates):
                return None
            values = {"trade_date": panel.dates[position]}
            for field in V8MarketPanel.FIELDS:
                value = float(panel.arrays[field][code_index, position])
                values[field] = value if np.isfinite(value) else None
            return values

        entry = day(date_index + 1)
        future = [day(position) or {} for position in range(date_index + 2, date_index + 2 + max(V8_HORIZONS))]
        return compute_path_labels(entry, future, costs=costs)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1"})


def _values_match(left: Any, right: Any, tolerance: float) -> bool:
    left_missing = left is None or pd.isna(left)
    right_missing = right is None or pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    if isinstance(right, (bool, np.bool_)):
        return str(left).lower() in ({"true", "1"} if bool(right) else {"false", "0"})
    if isinstance(right, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))
        except (TypeError, ValueError):
            return False
    return str(left).replace(".0", "") == str(right).replace(".0", "")


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()
