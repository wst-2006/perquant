from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP = "保留"
WATCH = "观察"
DOWNGRADE = "降级"


def build_package(
    factor_pool_version: str,
    training_report_path: str,
    convergence_report_path: str,
    registry_path: str,
    output_path: str,
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
            KEEP: [row for row in summary_rows if row.get("verdict") == KEEP],
            WATCH: [row for row in summary_rows if row.get("verdict") == WATCH],
            DOWNGRADE: [row for row in summary_rows if row.get("verdict") == DOWNGRADE],
        },
        "registry": registry,
    }
    Path(output_path).write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned ML Lab package.")
    parser.add_argument("--factor-pool-version", required=True)
    parser.add_argument("--training-report-path", default="reports/ml_lab/models_versioned_v2/model_comparison_report.json")
    parser.add_argument("--convergence-report-path", default="reports/ml_lab/convergence_versioned_v2/factor_convergence_details.json")
    parser.add_argument("--registry-path", default="reports/ml_lab/versioned_registry.json")
    parser.add_argument("--output-path", default="reports/ml_lab/execution_package_v1.json")
    args = parser.parse_args()
    package = build_package(args.factor_pool_version, args.training_report_path, args.convergence_report_path, args.registry_path, args.output_path)
    print(json.dumps(package, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
