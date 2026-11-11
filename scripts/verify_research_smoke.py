from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_METRICS = {"mean_net_return", "win_rate", "max_drawdown", "return_drawdown_ratio"}


class SmokeVerificationError(RuntimeError):
    pass


def verify_report(report_path: str | Path, summary_path: str | Path | None = None) -> dict:
    report_path = Path(report_path)
    if not report_path.exists():
        raise SmokeVerificationError(f"Report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("fold_count") or 0) <= 0:
        raise SmokeVerificationError("fold_count must be greater than 0")
    if not report.get("summary"):
        raise SmokeVerificationError("summary must not be empty")

    trained = [item for item in report.get("results", []) if item.get("status") == "trained"]
    targets = {item.get("target") for item in trained}
    missing_targets = {"stable", "aggressive"} - targets
    if missing_targets:
        raise SmokeVerificationError(f"Missing trained targets: {sorted(missing_targets)}")

    for target in ("stable", "aggressive"):
        target_rows = [item for item in trained if item.get("target") == target]
        if not target_rows:
            raise SmokeVerificationError(f"No trained rows for target={target}")
        metric_seen = False
        for row in target_rows:
            for top_name, metrics in (row.get("metrics") or {}).items():
                missing = REQUIRED_METRICS - set(metrics)
                if missing:
                    raise SmokeVerificationError(f"{target} {top_name} missing metrics: {sorted(missing)}")
                metric_seen = True
        if not metric_seen:
            raise SmokeVerificationError(f"No TopN metrics found for target={target}")

    if summary_path is not None and not Path(summary_path).exists():
        raise SmokeVerificationError(f"Summary not found: {summary_path}")

    return {
        "success": True,
        "report_path": str(report_path),
        "summary_path": str(summary_path) if summary_path else None,
        "fold_count": report.get("fold_count"),
        "targets": sorted(targets),
        "summary_rows": len(report.get("summary") or []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ML research smoke output before full GPU experiments.")
    parser.add_argument("--report-path", default="reports/ml_lab/research_harness_preflight/research_harness_report.json")
    parser.add_argument("--summary-path", default="reports/ml_lab/research_harness_preflight/research_harness_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = verify_report(args.report_path, args.summary_path)
    except SmokeVerificationError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
