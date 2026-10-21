import datetime
import os
import pandas as pd
from typing import Optional
from quant_research.registry.hashing import compute_file_sha256


class HoldoutQuarantineManager:
    """Enforces physical path separation and access audit logging for the pristine holdout test set."""

    def __init__(
        self,
        quarantine_dir: str,
        audit_log_path: str,
    ):
        self.quarantine_dir = os.path.abspath(quarantine_dir)
        self.audit_log_path = os.path.abspath(audit_log_path)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

    def deposit_holdout(self, df: pd.DataFrame, filename: str = "holdout_raw.csv") -> str:
        """Stores the holdout data in the quarantine directory and creates a lock checksum."""
        target_path = os.path.join(self.quarantine_dir, filename)
        df.to_csv(target_path, index=True)

        sha = compute_file_sha256(target_path)
        lockfile = f"{target_path}.sha256"
        with open(lockfile, "w", encoding="utf-8") as f:
            f.write(sha)

        self._log_access(
            action="DEPOSIT",
            filepath=target_path,
            sha256=sha,
            caller="Initial quarantine lock",
        )
        return target_path

    def load_holdout(self, caller: str, unlock_token: Optional[str] = None, filename: str = "holdout_raw.csv") -> pd.DataFrame:
        """Loads the quarantined holdout dataset, verifying checksum and recording single-read audit log."""
        target_path = os.path.join(self.quarantine_dir, filename)
        if not os.path.isfile(target_path):
            raise FileNotFoundError(f"Holdout file not found at {target_path}")

        lockfile = f"{target_path}.sha256"
        if not os.path.isfile(lockfile):
            raise PermissionError("Quarantine lockfile missing. Holdout integrity cannot be verified.")

        with open(lockfile, "r", encoding="utf-8") as f:
            expected_sha = f.read().strip()

        actual_sha = compute_file_sha256(target_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"HOLDOUT INTEGRITY BREACH: SHA256 mismatch!\nExpected: {expected_sha}\nActual: {actual_sha}"
            )

        self._log_access(
            action="READ_UNLOCKED",
            filepath=target_path,
            sha256=actual_sha,
            caller=caller,
            token=unlock_token,
        )

        return pd.read_csv(target_path, index_col=0, parse_dates=True)

    def _log_access(self, action: str, filepath: str, sha256: str, caller: str, token: Optional[str] = None):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_line = (
            f"[{timestamp}] ACTION={action} CALLER={caller} FILE={filepath} "
            f"SHA256={sha256} TOKEN={token or 'NONE'}\n"
        )
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
