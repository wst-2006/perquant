import json

import pandas as pd

from src.quant_trading.ml_lab.v8_labels import V8_LABEL_CONTRACT
from src.quant_trading.ml_lab.v8_research import V8ResearchConfig, V8ResearchHarness


def _dataset(tmp_path):
    root = tmp_path / "v8_dataset"
    root.mkdir()
    for day in range(1, 13):
        trade_date = f"202501{day:02d}"
        rows = []
        for stock in range(20):
            signal = ((stock * 3 + day) % 20) / 20
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"000{stock:03d}.SZ",
                    "industry": f"I{stock % 8}",
                    "stable_score": signal * 100,
                    "aggressive_score": (1 - signal) * 100,
                    "feature_return_20d": signal,
                    "feature_main_net_to_amount_sum_5d": signal * 0.5,
                    "label_entry_tradable": True,
                    "label_20d_end_date": f"202501{min(day + 1, 31):02d}",
                    "label_5d_net_return": signal * 0.08 - 0.02,
                    "label_10d_mfe_close": signal * 0.15,
                    "label_10d_mae_close": -0.05 + signal * 0.04,
                    "label_10d_hit_5pct": signal >= 0.4,
                    "label_20d_hit_8pct": signal >= 0.55,
                }
            )
        partition = root / f"trade_date={trade_date}"
        partition.mkdir()
        pd.DataFrame(rows).to_parquet(partition / "part-000.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"label_contract": V8_LABEL_CONTRACT, "source_sha256": "test"}),
        encoding="utf-8",
    )
    return root


def test_embargo_removes_labels_touching_validation():
    frame = pd.DataFrame(
        {
            "label_20d_end_date": ["20250105", "20250106"],
            "value": [1, 2],
        }
    )

    clean, audit = V8ResearchHarness.apply_embargo(frame, "20250106")

    assert clean["value"].tolist() == [1]
    assert audit["status"] == "pass"
    assert audit["embargoed_rows"] == 1


def test_cpu_ranker_smoke_writes_audited_report(tmp_path):
    result = V8ResearchHarness().run(
        _dataset(tmp_path),
        tmp_path / "output",
        V8ResearchConfig(
            train_windows=(6,),
            validation_days=2,
            step_days=2,
            models=("lightgbm",),
            device="cpu",
            n_estimators=5,
            max_folds=1,
        ),
    )

    assert result["summary"][0]["model"] == "lightgbm"
    assert result["summary"][0]["folds"] == 1
    assert result["model_selection_partition"]["status"] == "blocked"
    assert result["gate"]["promotion_allowed"] is False
    assert (tmp_path / "output" / "v8_research_report.json").exists()
    assert (tmp_path / "output" / "progress.jsonl").exists()
    assert (tmp_path / "output" / "partial_results.json").exists()
    progress_lines = (tmp_path / "output" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["event"] == "model_completed" for line in progress_lines)
    partial = json.loads((tmp_path / "output" / "partial_results.json").read_text(encoding="utf-8"))
    assert partial["status"] == "complete"


def test_model_selection_and_final_audit_are_chronologically_separate():
    rows = []
    for model in ("lightgbm", "xgboost"):
        for fold, day in enumerate(("20250101", "20250201", "20250301", "20250401", "20250501")):
            rows.append(
                {
                    "status": "trained",
                    "model": model,
                    "train_window": 120,
                    "validation_start": day,
                    "fold_index": fold,
                }
            )

    partition = V8ResearchHarness.partition_model_selection(
        rows,
        V8ResearchConfig(final_audit_fraction=0.4, min_final_audit_folds=2),
    )

    assert partition["status"] == "pass"
    assert partition["selection_validation_starts"] == ["20250101", "20250201", "20250301"]
    assert partition["audit_validation_starts"] == ["20250401", "20250501"]


def test_catboost_gpu_ranker_keeps_full_day_group_on_cpu():
    config = V8ResearchConfig(device="gpu")

    assert V8ResearchHarness.effective_backend("catboost", "rank", config).startswith("cpu_full_day_group")
    assert V8ResearchHarness.effective_backend("catboost", "classification", config) == "gpu"
    assert V8ResearchHarness.effective_backend("xgboost", "rank", config) == "gpu"
