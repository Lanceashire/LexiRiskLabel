"""Verify authorized local datasets and materialize both runtime layouts.

No data are downloaded.  Users must supply files they are authorized to use.
The exact registered SHA-256 digest is required unless explicitly overridden.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "uci": {
        "sha256": "8c05bc24646b74bec483f240be19bd21483170f029c5a7e30c6553a8ab452e91",
        "destinations": ["data/raw/uci/uci_default_credit_card.csv", "data/raw/uci_default_credit_card.csv"],
    },
    "hmeq": {
        "sha256": "dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa",
        "destinations": ["data/raw/hmeq/hmeq.csv", "data/raw/hmeq.csv"],
    },
    "ppdai": {
        "sha256": "07a2c70ad02cb858dfa18817a587f437c7ca1dc97c033360387f56cf177acdc2",
        "destinations": ["data/processed/ppdai/ppdai.csv", "data/phase3/ppdai.csv"],
    },
    "lendingclub": {
        "sha256": "2a2907dfeeb2b25a415e8802a2bc982a67fb08d10631a2b2a08aee47a103594d",
        "destinations": ["data/processed/lendingclub/lendingclub.csv", "data/phase3/lendingclub.csv"],
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in SPECS:
        parser.add_argument(f"--{dataset}", type=Path, required=True)
    parser.add_argument("--allow-hash-mismatch", action="store_true",
                        help="Materialize non-registered inputs and print a warning. Never use for paper parity claims.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing runtime copies.")
    args = parser.parse_args()
    for dataset, spec in SPECS.items():
        source = getattr(args, dataset).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{dataset}: {source}")
        actual = digest(source)
        if actual != spec["sha256"] and not args.allow_hash_mismatch:
            raise ValueError(f"{dataset}: SHA-256 mismatch; expected {spec['sha256']}, got {actual}")
        if actual != spec["sha256"]:
            print(f"WARNING {dataset}: non-registered hash {actual}")
        for relative in spec["destinations"]:
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not args.force:
                if digest(target) != actual:
                    raise FileExistsError(f"{target} exists with a different hash; pass --force intentionally")
                print(f"verified {target.relative_to(ROOT)}")
                continue
            shutil.copy2(source, target)
            print(f"materialized {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
