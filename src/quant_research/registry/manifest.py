import datetime
import json
import os
from typing import Dict, List, Optional
from quant_research.registry.hashing import compute_directory_manifest, compute_file_sha256


class ExperimentManifestRegistry:
    """Manages experiment provenance, code/data hashes, and pipeline parameters."""

    def __init__(self, registry_file: str):
        self.registry_file = registry_file
        self.records: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []
        else:
            self.records = []

    def record_run(
        self,
        experiment_id: str,
        stage: str,
        parameters: Dict,
        code_dir: Optional[str] = None,
        data_files: Optional[List[str]] = None,
        metrics: Optional[Dict] = None,
        status: str = "SUCCESS",
    ) -> Dict:
        """Records an execution step with cryptographic fingerprints."""
        data_hashes = {}
        if data_files:
            for df in data_files:
                if os.path.isfile(df):
                    data_hashes[os.path.basename(df)] = compute_file_sha256(df)

        code_manifest = {}
        if code_dir and os.path.isdir(code_dir):
            code_manifest = compute_directory_manifest(
                code_dir, include_extensions=[".py", ".yaml", ".json"]
            )

        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "stage": stage,
            "status": status,
            "parameters": parameters,
            "data_hashes": data_hashes,
            "code_manifest_summary": {
                "file_count": len(code_manifest),
                "combined_hash": compute_file_sha256(self.registry_file)
                if os.path.isfile(self.registry_file)
                else None,
            },
            "metrics": metrics or {},
        }

        self.records.append(entry)
        self._save()
        return entry

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.registry_file)), exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
