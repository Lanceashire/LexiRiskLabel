# Boundary-Condition Analysis

1. **Exposure boundary.** Unit exposure (PPDai) produces near-indistinguishable policies; heterogeneous loan amount exposure (LendingClub, CV=0.5796) produces measurable separation.
2. **Budget boundary.** LendingClub policy ranges increase with rho: FDRatio range 0.049789, 0.078999, 0.125615, and 0.199837 for rho 0.025, 0.05, 0.10, and 0.20 respectively.
3. **Statistical boundary.** Separation is not equivalent to superiority: all primary FAVE pairwise Holm-adjusted tests remain non-significant at the tested rhos.
4. **Provenance boundary.** PPDai outcome-adjacent feature provenance is not fully decision-time provable; those results are not treated as clean causal evidence.
5. **Operational boundary.** FAVE linear algebra is the dominant compute cost. GPU Hessian construction, Cholesky, and solves were parity-validated; this is an exact engineering optimization, not a method change.