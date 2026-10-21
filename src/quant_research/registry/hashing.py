import hashlib
import os
from typing import Dict, List, Optional


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_directory_manifest(
    directory: str,
    include_extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generates a mapping of relative paths to SHA-256 hashes for a directory tree."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")

    exclude_dirs = exclude_dirs or [
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".venv",
        "venv",
    ]
    manifest: Dict[str, str] = {}

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if include_extensions:
                if not any(f.endswith(ext) for ext in include_extensions):
                    continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, directory).replace("\\", "/")
            manifest[rel_path] = compute_file_sha256(full_path)

    return manifest
