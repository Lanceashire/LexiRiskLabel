# Phase 1 protocol

Treatments are exactly Oracle, NoExplore, RandomExplore-2%, RandomExplore-5%, and RandomExplore-10%. No FAVE, CountOnly, or risk-budget optimization is used.

Each dataset/seed shares one fixed 25% oracle test set, one D0 (20% of the non-test stream), one applicant ordering, and one exact ten-way batch partition. Models and preprocessing are fit only on currently visible labels. Random exploration samples only the current round rejected set with deterministic seed seed*100000 + round*100 + rate_code. Unselected rejected applicants leave the episode permanently.

Formal seeds are 0..19 for both UCI Default of Credit Card Clients and HMEQ. Phase 1 stops after mechanism validation; no Phase 2 execution is authorized by this protocol.
