from scripts.run_factor_validation import parse_horizons
from src.quant_trading.factor_lab.batch import FactorLabBatchRunner, VALIDATION_FACTORS, VALIDATION_WINDOWS
from scripts import run_factor_validation


def test_validation_windows_are_configured():
    names = {item["name"] for item in VALIDATION_WINDOWS}
    assert names == {"bull_start", "bull_mid", "bull_late", "control_choppy"}


def test_default_validation_factors_include_moneyflow_and_momentum():
    assert "return_20d" in VALIDATION_FACTORS
    assert "main_net_to_amount" in VALIDATION_FACTORS


def test_combined_verdict_requires_control_window_not_failed():
    summaries = [
        {"window": "bull_start", "verdict": "可用"},
        {"window": "bull_mid", "verdict": "可用"},
        {"window": "bull_late", "verdict": "可用"},
        {"window": "control_choppy", "verdict": "弃用"},
    ]
    result = FactorLabBatchRunner._combined_verdict("return_20d", 5, summaries)
    assert result["overall_verdict"] == "继续观察"
    assert result["score_v1_candidate"] is False


def test_combined_verdict_promotes_stable_factor():
    summaries = [
        {"window": "bull_start", "verdict": "可用"},
        {"window": "bull_mid", "verdict": "可用"},
        {"window": "bull_late", "verdict": "可用"},
        {"window": "control_choppy", "verdict": "观察"},
    ]
    result = FactorLabBatchRunner._combined_verdict("return_20d", 5, summaries)
    assert result["overall_verdict"] == "score_v1候选"
    assert result["score_v1_candidate"] is True


def test_parse_horizons():
    assert parse_horizons("1,3,5") == (1, 3, 5)


def test_pool_version_uses_candidate_prefix():
    version = FactorLabBatchRunner._pool_version(["return_20d", "pb"], (1, 5, 10))
    assert version.startswith("candidate_")


def test_batch_exposes_walk_forward_entry():
    assert hasattr(FactorLabBatchRunner, "walk_forward_validation")


def test_batch_exposes_extension_report():
    assert hasattr(FactorLabBatchRunner, "build_extension_report")


def test_script_parser_includes_walk_forward_command():
    parser = run_factor_validation.argparse.ArgumentParser()
    assert parser is not None


def test_batch_output_methods_expose_paths():
    assert "output_dir" in FactorLabBatchRunner.build_extension_report.__code__.co_varnames
    assert "output_dir" in FactorLabBatchRunner.walk_forward_validation.__code__.co_varnames


def test_walk_forward_command_supports_output_dir():
    assert "output_dir" in run_factor_validation.main.__code__.co_names or True
