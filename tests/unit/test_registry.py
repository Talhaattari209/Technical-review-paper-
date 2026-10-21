import os
import pandas as pd
import pytest
from quant_research.registry.hashing import compute_file_sha256
from quant_research.registry.holdout import HoldoutQuarantineManager
from quant_research.registry.manifest import ExperimentManifestRegistry


def test_hashing_and_manifest(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Quantitative Governance v3.0", encoding="utf-8")

    sha = compute_file_sha256(str(test_file))
    assert len(sha) == 64
    assert isinstance(sha, str)

    reg_file = tmp_path / "registry.json"
    registry = ExperimentManifestRegistry(str(reg_file))
    entry = registry.record_run(
        experiment_id="EXP_001",
        stage="TEST",
        parameters={"lookback": 20},
        data_files=[str(test_file)],
        metrics={"sharpe": 1.2},
    )
    assert entry["experiment_id"] == "EXP_001"
    assert "sample.txt" in entry["data_hashes"]
    assert os.path.exists(str(reg_file))


def test_holdout_quarantine_tamper_detection(tmp_path):
    quarantine_dir = tmp_path / "holdout"
    audit_log = tmp_path / "audit.log"
    manager = HoldoutQuarantineManager(str(quarantine_dir), str(audit_log))

    df = pd.DataFrame({"close": [100.0, 101.0, 102.5]}, index=pd.date_range("2026-01-01", periods=3))
    deposited_path = manager.deposit_holdout(df, "test_holdout.csv")
    assert os.path.exists(deposited_path)

    # Valid read
    loaded_df = manager.load_holdout("test_runner", unlock_token="AUTH-01", filename="test_holdout.csv")
    assert len(loaded_df) == 3

    # Tamper with file
    with open(deposited_path, "a") as f:
        f.write("tampered,data\n")

    with pytest.raises(ValueError, match="HOLDOUT INTEGRITY BREACH"):
        manager.load_holdout("malicious_runner", filename="test_holdout.csv")

    assert os.path.exists(str(audit_log))
