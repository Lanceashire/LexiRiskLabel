# Controlled mechanism conditions

This preregistered synthetic extension tests when information-oriented feedback
selection can repay its count and predicted-risk cost. It complements the
four-dataset dynamic study with a one-step common-state design.

Run in the validated environment:

```text
$env:PHASE2_USE_GPU='1'
$env:PHASE2_GPU_DEVICE='cuda:0'
D:\envs_dirs\no_1\python.exe experiments\mechanism_conditions\run_mechanism_conditions.py --seeds 2999 --members 3 --output results\mechanism_conditions_smoke_20260908
D:\envs_dirs\no_1\python.exe experiments\mechanism_conditions\run_mechanism_conditions.py --output results\mechanism_conditions_primary_20260908
D:\envs_dirs\no_1\python.exe experiments\mechanism_conditions\analyze_mechanism_conditions.py results\mechanism_conditions_primary_20260908 --output results\mechanism_conditions_primary_20260908_analysis
```

An existing output directory requires `--resume`; existing seed files are never
overwritten. Generated synthetic observations are deterministic from the seed
and are not stored as a redistributed dataset. See `PREREGISTRATION.md` for the
frozen hypotheses and claim boundaries.
