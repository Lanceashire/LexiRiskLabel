"""Read-only verification for the GitHub publication package."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "README.md", "DATA_AVAILABILITY.md", "REPRODUCIBILITY.md", "RESULTS.md",
    "LICENSE", "LICENSES_AND_DATA_TERMS.md", "CITATION.cff", "requirements.txt",
    "environment.yml", "results/ccfb_extension/CCFB_extension_report.md",
    "results/ccfb_extension/01_kstar_ablation_raw.csv",
    "results/ccfb_extension/03_budget_sweep_raw.csv",
    "results/ccfb_extension/05_budget_reliability_raw.csv",
    "results/ccfb_extension/08_feedback_decision_statistics.csv",
]
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "logs", "profiles", "archive"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    forbidden = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts)
    ]
    dataset_files = [path for area in (ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "phase3")
                     if area.exists() for path in area.rglob("*") if path.is_file()]
    oversized = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 100 * 1024 * 1024]
    raw_root = ROOT / "results" / "ccfb_extension" / "raw"
    formal = [path for experiment in ("reference", "kstar_ablation", "budget_sweep")
              for path in (raw_root / experiment).rglob("*.csv")]
    manifest_path = ROOT / "PUBLICATION_MANIFEST.csv"
    manifest_failures = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = ROOT / row["path"]
                if not path.is_file() or digest(path) != row["sha256"]:
                    manifest_failures.append(row["path"])
    print(f"required_missing={len(missing)}")
    print(f"forbidden_files={len(forbidden)}")
    print(f"third_party_dataset_files={len(dataset_files)}")
    print(f"files_over_100MiB={len(oversized)}")
    print(f"formal_ccfb_raw_files={len(formal)} (expected 1120)")
    print(f"publication_manifest_failures={len(manifest_failures)}")
    if missing or forbidden or dataset_files or oversized or len(formal) != 1120 or manifest_failures:
        raise SystemExit(1)
    print("PUBLICATION_AUDIT_STATUS=PASS")


if __name__ == "__main__":
    main()
