import zipfile

from scripts.package_gpu_experiment import build_package


def test_gpu_package_excludes_secrets_and_includes_required_files(tmp_path):
    output_path = tmp_path / "gpu_package.zip"

    manifest = build_package(output_path)

    assert manifest["contains_env"] is False
    assert manifest["contains_database"] is False
    assert output_path.exists()
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
    assert ".env" not in names
    assert "requirements-gpu.txt" in names
    assert "scripts/run_gpu_full_baseline.sh" in names
    assert "scripts/run_ml_lab.py" in names
    assert "reports/ml_lab/training_frame_full.csv" in names
    assert "reports/ml_lab/experiment_manifest.json" in names


def test_gpu_package_accepts_v2_training_frame_and_split_manifest(tmp_path):
    training_frame = tmp_path / "training_frame_v2.csv"
    manifest = tmp_path / "manifest.json"
    split_manifest = tmp_path / "split.json"
    training_frame.write_text("trade_date,ts_code,label_tradable,label_net_return,label_stable_positive,label_top10,feature_x\n20250101,000001.SZ,1,0.01,1,1,0.5\n", encoding="utf-8")
    manifest.write_text('{"kind":"experiment"}', encoding="utf-8")
    split_manifest.write_text('{"holdout_dates":["20250102"]}', encoding="utf-8")
    output_path = tmp_path / "gpu_package_v2.zip"

    package_manifest = build_package(
        output_path,
        training_frame_path=training_frame,
        manifest_path=manifest,
        extra_paths=[split_manifest],
    )

    assert package_manifest["contains_env"] is False
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
    assert training_frame.name in names
    assert manifest.name in names
    assert split_manifest.name in names
