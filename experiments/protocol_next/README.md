# Next-protocol development diagnostics

Implements the user-supplied `experiment_protocol_zh.md` against the frozen
`45a2d41d403ce0587d5434f564abb76ac7eaaf48` research-v2 development evidence.
Existing code and prior results are preserved.

Completed scope:

- E0: role/label boundaries, immutable source identity, cache and action identity,
  actual backend, release timing, inaccessible CORR support, mathematical controls.
- E1: 18 registered MinRisk anchors (three datasets, two seeds, rounds 2/5/9),
  eight selector types with five fixed random draws; the SAME selected set goes
  to LR and full raw-feature GPU XGBoost.
- Conditional E2b: all affordable pairs and a shared-geometry construction x
  exchange factorial. The common exchange neighborhood is one 1-to-1 pass with
  eight removal probes; pair construction is evaluated independently.
- Conditional E2c: all six registered round-5 anchors, D-only 20-tree/depth-2
  frozen leaf features; compare raw128 vs leaf128, and separately raw128 vs
  raw512. Full XGBoost always regrows trees on original standardized features.
- Conditional E1b: sixteen distinct feasible sets at each of the six round-5
  anchors, fixed before revealing labels, selected on independent V_search and
  evaluated on V_eval. This extra-validation-label reference is diagnostic only.

- E2d: the six fixed round-5 anchors, deduplicated exact counts at fractions
  0.6/0.8/1.0 of Kstar, MinRisk-k/IG-k/Vopt-k and OrdinaryOnly. One random
  one-for-one swap per IG/Vopt target is accepted only within 1% of B in risk;
  unmatched targets remain not estimable. This is a local composition diagnostic.

No new hyperparameter selection, seed expansion, CORR/GATE scale-up, or 720
confirmatory ten-round episodes. The full E2a grid is not run.
The analysis records those limits and gives a continue/stop decision for each route.

## Reproduce

Use the existing project-local Python environment with the locked numerical
dependencies and actual NVIDIA GPU access. Full local v2 anchors and authorized
`data/research_v2` inputs are required; the public aggregate-only bundle cannot
reconstruct missing individual records.

```bash
python experiments/protocol_next/run_e1.py --output /new/diagnostic-output
python experiments/protocol_next/run_e2b.py --base /new/diagnostic-output
python experiments/protocol_next/run_e2c.py --base /new/diagnostic-output
python experiments/protocol_next/run_e1b.py --base /new/diagnostic-output
python experiments/protocol_next/analyze_next.py --base /new/diagnostic-output
python experiments/protocol_next/run_e2d.py --base /new/diagnostic-output
python experiments/protocol_next/analyze_e2d.py --base /new/diagnostic-output
```

E2b/E2c are conditional, not automatic confirmation stages. Read the E1 outcomes
and record why their activation is justified. Each conditional runner writes its
source fingerprint, parent identity and fixed scope before execution.

`protocol.json` is the effective E0/E1 configuration. `e2c_config.json` locks the
representation and reference-size contrasts; `e2d_config.json` fixes the count
and risk-matched swap rules before E2d evaluation. `FutureProtocol` separates planning
horizon from actual future-label release and holds a common future size/capacity/
risk budget; this is an audited interface for later E4, not a rerun of old GATE.

All prediction arrays, individual selections, true errors and raw pair-Gamma logs
stay under the output's `private/` directory. Public JSONL/CSV tables contain
aggregate outcomes and auditable hashes only. Fitting caches do not add independent
replications; random draws average within states, and three states average within
each trajectory. Two development seeds do not justify confirmatory significance.
