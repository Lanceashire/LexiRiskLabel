#!/usr/bin/env bash
set -euo pipefail
experiment_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${EXPERIMENT_PYTHON:-python3}"
experiment_output="${1:?Provide a NEW output directory; frozen results are never overwritten}"
if [[ -e "$experiment_output" ]]; then
  echo 'Output already exists; choose a new directory.' >&2
  exit 2
fi
mkdir -p "$experiment_output"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$experiment_python" "$experiment_here/run_mechanisms.py" --output "$experiment_output/final_mechanisms"
"$experiment_python" "$experiment_here/run_real.py" --output "$experiment_output/final_real" --protocol fixed
"$experiment_python" "$experiment_here/run_real.py" --output "$experiment_output/final_real" --protocol correction
"$experiment_python" "$experiment_here/run_focused.py" --output "$experiment_output/focused"
"$experiment_python" "$experiment_here/verify_results.py" --base "$experiment_output"
MPLCONFIGDIR="$experiment_output/.matplotlib" "$experiment_python" "$experiment_here/analyze.py" --base "$experiment_output" --output "$experiment_output/analysis"
