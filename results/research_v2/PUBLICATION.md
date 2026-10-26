# Published development results

This publication contains the research v2 code, report, figures, aggregate seed
comparisons, aggregate per-round trajectories, configurations, dataset provenance,
and verification summaries. It includes all positive and negative conclusions.

The complete local experiment archive also retains per-applicant selection,
probability and correction records, plus `test_predictions.npz` arrays containing
sample IDs, targets and predictions. These individual-level artifacts are omitted
from GitHub. Original third-party input data and private planning conversations
are also omitted. A fresh authorized local run regenerates the individual-level
artifacts for `verify_results.py`; the published verification summary describes
the complete local run, not a revalidation from aggregate files alone.

Final development scope: 84 real-data episodes, two seeds per dataset, ten rounds,
one risk budget; six small synthetic environments with three development seeds.
These results do not establish confirmatory superiority of the proposed methods.

- [Report](analysis/实验汇总.md)
- [Aggregate method comparison](analysis/method_summary.csv)
- [Paired descriptive differences](analysis/paired_descriptive_contrasts.csv)
- [Local-run verification summary](verification.json)
- [Code and reproduction](../../experiments/research_v2/README.md)
