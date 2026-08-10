"""Build the repository-level SHA-256 publication manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PUBLICATION_MANIFEST.csv"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path != OUTPUT
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["path", "sha256", "bytes"])
        for path in files:
            writer.writerow([
                path.relative_to(ROOT).as_posix(),
                digest(path),
                path.stat().st_size,
            ])
    print(f"publication_manifest_entries={len(files)}")
    print(f"publication_manifest={OUTPUT}")


if __name__ == "__main__":
    main()
