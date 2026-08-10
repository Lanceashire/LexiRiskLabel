# Phase 1 integrity audit

dataset                                    check  passed                                                                     detail
    uci rows_20_seeds_x_5_treatments_x_11_rounds    True                                                                       1100
    uci                            seeds_0_to_19    True     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    uci                         treatments_exact    True [NoExplore, Oracle, RandomExplore-10%, RandomExplore-2%, RandomExplore-5%]
    uci                           rounds_0_to_10    True                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    uci                       runtime_invariants    True                                                                   all rows
    uci       same_batch_sizes_across_treatments    True                                                                   per seed
    uci         forbidden_fave_treatments_absent    True                                                Phase 1 treatment isolation
   hmeq rows_20_seeds_x_5_treatments_x_11_rounds    True                                                                       1100
   hmeq                            seeds_0_to_19    True     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
   hmeq                         treatments_exact    True [NoExplore, Oracle, RandomExplore-10%, RandomExplore-2%, RandomExplore-5%]
   hmeq                           rounds_0_to_10    True                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   hmeq                       runtime_invariants    True                                                                   all rows
   hmeq       same_batch_sizes_across_treatments    True                                                                   per seed
   hmeq         forbidden_fave_treatments_absent    True                                                Phase 1 treatment isolation

All checks passed: True
