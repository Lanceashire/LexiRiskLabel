# Reproduction preflight; deliberately does not launch a model run.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    required = [ROOT / "DATASET_REGISTRY.csv", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "ARTIFACT_HASHES.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Preflight failed: " + ", ".join(missing))
    print("Preflight only: no experiment has been started.")
    print("Canonical root:", ROOT)

if __name__ == "__main__":
    main()
