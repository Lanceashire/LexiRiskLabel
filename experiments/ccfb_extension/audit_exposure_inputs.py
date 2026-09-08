"""Audit the exact study inputs used by the CCFB extension.

The audit is read-only.  It reports hashes and exposure validity without
silently coercing invalid observations into zero-cost feedback.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "HMEQ": (ROOT / "data/raw/hmeq/hmeq.csv", "LOAN"),
    "PPDai": (ROOT / "data/processed/ppdai/ppdai.csv", "exposure"),
    "LendingClub": (
        ROOT / "data/processed/lendingclub/lendingclub.csv",
        "exposure",
    ),
}
EXPECTED_HASHES = {
    "HMEQ": "dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa",
    "PPDai": "07a2c70ad02cb858dfa18817a587f437c7ca1dc97c033360387f56cf177acdc2",
    "LendingClub": "2a2907dfeeb2b25a415e8802a2bc982a67fb08d10631a2b2a08aee47a103594d",
}


def audit_one(dataset: str, path: Path, exposure_col: str) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    raw = frame[exposure_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    finite_numeric = numeric.dropna().to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    report = {
        "dataset": dataset,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "exposure_column": exposure_col,
        "raw_missing": int(raw.isna().sum()),
        "non_numeric": int((raw.notna() & numeric.isna()).sum()),
        "nonfinite": int((~np.isfinite(finite_numeric)).sum()),
        "negative": int((numeric < 0).sum()),
        "zero": int((numeric == 0).sum()),
        "positive": int((numeric > 0).sum()),
        "minimum": float(finite.min()),
        "p01": float(finite.quantile(0.01)),
        "median": float(finite.median()),
        "p99": float(finite.quantile(0.99)),
        "maximum": float(finite.max()),
    }
    report["expected_sha256"] = EXPECTED_HASHES[dataset]
    report["hash_matches_registry"] = report["sha256"] == EXPECTED_HASHES[dataset]
    report["strictly_positive_finite"] = not any(
        report[key] for key in ("raw_missing", "non_numeric", "nonfinite", "negative", "zero")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    reports = [audit_one(name, *spec) for name, spec in INPUTS.items()]
    if not all(r["hash_matches_registry"] and r["strictly_positive_finite"] for r in reports):
        raise SystemExit("Exposure input audit failed")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "EXPOSURE_INPUT_AUDIT.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pd.DataFrame(reports).to_csv(args.output_dir / "EXPOSURE_INPUT_AUDIT.csv", index=False)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
