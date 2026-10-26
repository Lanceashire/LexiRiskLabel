#!/usr/bin/env bash
set -euo pipefail
experiment_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_repo="$(cd -- "$experiment_here/../.." && pwd)"
experiment_python="${EXPERIMENT_PYTHON:-python3}"
experiment_cache="${1:?Provide a local source-data cache directory}"
mkdir -p "$experiment_cache"
curl -fL --max-time 180 https://raw.githubusercontent.com/sassoftware/sas-viya-dmml-pipelines/master/data/hmeq.csv -o "$experiment_cache/hmeq_sas.csv"
curl -fL --max-time 180 https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip -o "$experiment_cache/uci.zip"
for experiment_year in 2014 2015; do
  curl -fL --max-time 180 "https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/LC_Loan_sample_${experiment_year}.csv" -o "$experiment_cache/LC_Loan_sample_${experiment_year}.csv"
done
"$experiment_python" - "$experiment_cache" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys
cache=Path(sys.argv[1]);dest=cache/'uci_source';dest.mkdir(exist_ok=True)
with ZipFile(cache/'uci.zip') as z:
    entry='default of credit card clients.xls'
    (dest/entry).write_bytes(z.read(entry))
PY
if [[ -e "$experiment_repo/data/research_v2" ]]; then
  echo 'Data destination already exists; retain it and inspect provenance before replacing anything.' >&2
  exit 2
fi
"$experiment_python" "$experiment_here/prepare_data.py" --source "$experiment_cache" --output "$experiment_repo/data/research_v2"
