# Mechanism experiments: consolidation and writing verification

2026-11-02. This release follows the user's experiment roadmap, with prospective
generator/precision amendments retained. It completes experiment consolidation
and verifies how the evidence can be written. It contains **no full manuscript,
anonymous submission PDF, or formal submission**.

Read [the Chinese consolidation report](research_consolidation.zh.txt),
[writing verification](submission_writing_verification.zh.txt), and
[item-by-item roadmap acceptance](original_roadmap_acceptance.csv).
The [self-contained review ZIP](review_package_20261102.zip) contains protocols,
frozen sources, world-level risks, aggregate real-data bootstrap records,
audits, figures, engineering entry points, and a portable independent verifier.

## Seven LR primary endpoints

Positive values follow each registered contrast. Multiplicity is controlled
within the allocated E1/E2/E3 families (.025/.015/.010); XGB remains secondary.

| Study / endpoint | Estimate | Simultaneous interval | Decision |
|---|---:|---:|---|
| E1 original-history J | +0.000459164 | [0.000368743, 0.000549585] | practical positive |
| E1 history × O-dose interaction | +0.000154068 | [0.000085366, 0.000222771] | practical magnitude unresolved |
| E1 broad-history J | +0.000010033 | [0.000004363, 0.000015704] | within practical band |
| E2 quantity QM | +0.000450913 | [0.000340469, 0.000561356] | practical positive |
| E2 finite matched CM | +0.001860219 | [0.001588461, 0.002131976] | practical positive, matched subgroup |
| E3 ten-round J | +0.000398321 | [-0.000016023, 0.000800648] | unresolved |
| E3 history interaction H | +0.000016644 | [-0.000028651, 0.000061753] | within practical band |

E1/E2 have 512 independent confirmation worlds each, excluding 32 pilots each.
E2 CM uses 469 jointly matchable worlds; the 43 remaining worlds are retained
in the matching ledger. CM concerns a fixed finite matching procedure, not
global optimality. E3 uses a previously unused **older LC2012 cohort**, 20 fixed
seeds and 1,200 test groups; it is neither future-calendar nor new-institution
validation. Its uncertainty is conditional on the fitted models. Thresholds
are 0.0001 for synthetic endpoints and 0.000630328 for E3.

## What the closure adds

The exact identity J = E_PA − E_MinRisk resolves an apparent sign conflict:
with more ordinary feedback, PA's absolute exploration gain decreases, while
its relative advantage increases because MinRisk's gain decreases more.
This is an accounting explanation, not a new primary test or pure causal
reoptimization effect. Fixed D-only is not a frozen copy of the PA policy.

The clean E0 engineering rerun records 380 full resource/loss rows and 380
lock/release pairs; the 202 split/prediction NPZ comparisons are bitwise
identical. The public schema reports expected and realized loss separately.
Its 120 realized-over-budget state records are not 120 independent applicants
or a new confirmatory violation-rate estimate.

All five conditional extension modules were reviewed. None requires new
training for the claims retained here; future single-factor causal attribution
or deployable diagnostic claims would need a new design. Two unresolved
primary results and unmet original precision targets remain explicit.

## Provenance and use

The unavailable external 16-world bridge implementation was **not recovered**.
The authorized prospective v2 generator and precision scope were frozen before
confirmation; original M1–M5 evidence remains unchanged. Completion means
execution and audits within that explicit scope, not universal confirmation
or achievement of every originally desired precision target.

The archive includes its original v2 subpackage byte-for-byte. Historical
publication=false/pending fields inside are pre-publication snapshots;
the GitHub commit and separate local publication receipt record this upload.
No real credit rows, individual labels, individual predictions, private
entity-loss files or private planning conversation are included. Raw arrays
under synthetic engineering examples are generated data.

After installing the pinned dependencies, run `python verify_consolidation.py`
from the extracted ZIP. It verifies all SHA256 entries and independently
reconstructs all 14 primary/secondary endpoint statistics and the fixed-action
identity. It does not regenerate real individual losses from aggregates.
For full private E0 replay, use the documented `closure/e0_public/run_public.py`;
without `--reference`, equality to old private predictions is marked untested.

Prior work in Fisher experimental design, informational substitutes, EPIG,
loss-driven Bayesian active learning, logged active learning and verification
latency provides the foundation. The writing guide limits proposed contributions
to tested history conditions, resource-controlled comparisons and transfer
boundaries. No priority claim or acceptance guarantee is made.
