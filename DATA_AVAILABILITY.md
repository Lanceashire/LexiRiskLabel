# Data Availability and Provenance

## Distribution policy

This repository intentionally excludes third-party raw and processed datasets.
Research use or public availability at a source does not automatically grant
permission to redistribute a copy under this repository's MIT license.

The repository therefore provides:

- source and provenance notes;
- target and exposure definitions;
- expected sample counts;
- SHA-256 digests of the exact study inputs;
- feature and leakage audits where available;
- a local layout materialization utility.

Users must obtain each dataset from a source they are authorized to use and
comply with its original terms.

## Registered study inputs

| Dataset | Evidence role | Rows | Study input SHA-256 | Target / exposure treatment |
|---|---|---:|---|---|
| UCI Default Credit Card | Development benchmark | 30,000 | `8c05bc24646b74bec483f240be19bd21483170f029c5a7e30c6553a8ab452e91` | Default target; unit-exposure risk proxy |
| HMEQ | Development monetary validation | 5,960 | `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa` | `BAD=1`; `PD_hat × LOAN`, fixed `LGD=1` |
| PPDai | Exploratory external boundary | 55,596 | `07a2c70ad02cb858dfa18817a587f437c7ca1dc97c033360387f56cf177acdc2` | Processed `label`; exposure proxy; lower-confidence evidence |
| LendingClub | Primary external validation | 16,566 | `2a2907dfeeb2b25a415e8802a2bc982a67fb08d10631a2b2a08aee47a103594d` | Mature Charged Off vs Fully Paid cohort; `loan_amnt` exposure proxy |

These hashes identify the exact processed inputs used by the study.  A file
from a similarly named dataset is not necessarily equivalent.

## Source notes

### UCI Default Credit Card

Obtain the Default of Credit Card Clients dataset from the UCI Machine
Learning Repository or another source whose terms permit your use.  Normalize
the target to the committed schema and verify the registered processed hash.

### HMEQ

The study uses the 5,960-row HMEQ table with target `BAD` and amount field
`LOAN`.  The repository does not assert redistribution rights for a particular
download copy.  Obtain an authorized copy and verify the hash above.

### PPDai

The frozen provenance records the Figshare source file `26049578`:

```text
https://ndownloader.figshare.com/files/26049578
```

The original downloaded object and the normalized study CSV are not committed.
PPDai remains explicitly lower-confidence exploratory evidence.

### LendingClub

The frozen provenance records public source samples from:

```text
https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/
```

The study uses a leakage-audited mature cohort and treats `loan_amnt` as an
exposure proxy.  Source samples and the processed cohort are not redistributed.

## Expected local layout

The CCFB extension expects:

```text
data/raw/uci/uci_default_credit_card.csv
data/raw/hmeq/hmeq.csv
data/processed/ppdai/ppdai.csv
data/processed/lendingclub/lendingclub.csv
```

Frozen Phase 1--3 runners retain their historical compatibility paths:

```text
data/raw/uci_default_credit_card.csv
data/raw/hmeq.csv
data/phase3/ppdai.csv
data/phase3/lendingclub.csv
```

Use `scripts/data_preparation/materialize_data_layout.py` to verify one local
copy per dataset and create both layouts.  The utility refuses hash mismatches
unless the user explicitly overrides that safeguard.

## Exposure integrity audit

The exact registered HMEQ, PPDai, and LendingClub study inputs were audited
for raw missing values, nonnumeric coercions, nonfinite values, negatives, and
zeros. All five counts are zero across 78,122 records, and all three file
digests match the registry. The machine-readable report is stored at
`results/ccfb_extension/exposure_hardening/EXPOSURE_INPUT_AUDIT.csv` and is
also copied into the canonical PaperOrchestra workspace.

The current CCFB extension loader requires strictly positive finite exposure
and raises an error on invalid input. This hardening does not change results on
the registered files: eight GPU-backed confirmation runs reproduce their
pre-hardening counterparts to numerical tolerance. Historical frozen runners
remain provenance artifacts; new CCFB runs use the fail-closed loader.

## Metadata included in this repository

See:

- `DATASET_REGISTRY.csv`;
- `data/metadata/dataset_hashes.csv`;
- `data/metadata/provenance/PHASE3_SOURCES.md`;
- `data/metadata/feature_audits/`;
- `data/metadata/leakage_audits/`;
- `data/metadata/provenance/TARGET_MAPPING_LENDINGCLUB.csv`.

## 中文说明

仓库不重新分发第三方原始或处理后数据。请从有权使用的来源自行取得数据，按照上表核对 SHA-256，并运行数据布局工具。代码的 MIT License 不自动覆盖第三方数据。
