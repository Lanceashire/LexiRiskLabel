# Phase 1 statistical report

All comparisons are paired by seed (20 seeds per dataset). Wilcoxon signed-rank tests use Holm correction within each dataset across the reported comparisons and metrics. Bootstrap confidence intervals use 10,000 fixed-seed resamples of paired seed differences.

## UCI significant or near-primary results

                    comparison            metric  mean_diff  bootstrap_ci_low  bootstrap_ci_high  wilcoxon_p  holm_p  wins  ties  losses
           NoExplore_vs_Oracle            recall    0.00337          -0.00067            0.00810     0.21221 1.00000    12     1       7
           NoExplore_vs_Oracle balanced_accuracy    0.00054          -0.00111            0.00243     0.78413 1.00000    10     0      10
           NoExplore_vs_Oracle       visible_tdr   -0.04724          -0.04768           -0.04679     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle          fd_ratio   -0.21383          -0.21573           -0.21195     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle             brier    0.00056           0.00039            0.00073     0.00002 0.00069    18     0       2
           NoExplore_vs_Oracle               ece    0.00128           0.00038            0.00229     0.01923 0.53854    14     0       6
           NoExplore_vs_Oracle           roc_auc    0.00013          -0.00032            0.00053     0.29425 1.00000    13     0       7
 RandomExplore-2%_vs_NoExplore            recall   -0.00379          -0.00622           -0.00138     0.01758 0.51544     7     1      12
 RandomExplore-2%_vs_NoExplore balanced_accuracy   -0.00167          -0.00275           -0.00058     0.01718 0.51544     6     0      14
 RandomExplore-2%_vs_NoExplore       visible_tdr    0.00098           0.00084            0.00111     0.00000 0.00008    20     0       0
 RandomExplore-2%_vs_NoExplore          fd_ratio    0.00441           0.00382            0.00504     0.00000 0.00008    20     0       0
 RandomExplore-2%_vs_NoExplore             brier    0.00002          -0.00008            0.00011     0.59582 1.00000    11     0       9
 RandomExplore-2%_vs_NoExplore               ece    0.00024          -0.00065            0.00114     0.64766 1.00000    12     0       8
 RandomExplore-2%_vs_NoExplore           roc_auc   -0.00019          -0.00054            0.00018     0.16496 1.00000     5     0      15
 RandomExplore-5%_vs_NoExplore            recall   -0.00270          -0.00600            0.00046     0.24320 1.00000     9     1      10
 RandomExplore-5%_vs_NoExplore balanced_accuracy   -0.00108          -0.00237            0.00014     0.24549 1.00000     9     0      11
 RandomExplore-5%_vs_NoExplore       visible_tdr    0.00254           0.00238            0.00270     0.00000 0.00008    20     0       0
 RandomExplore-5%_vs_NoExplore          fd_ratio    0.01151           0.01079            0.01224     0.00000 0.00008    20     0       0
 RandomExplore-5%_vs_NoExplore             brier    0.00001          -0.00008            0.00009     0.92728 1.00000    12     0       8
 RandomExplore-5%_vs_NoExplore               ece    0.00099           0.00018            0.00180     0.01531 0.47468    15     0       5
 RandomExplore-5%_vs_NoExplore           roc_auc   -0.00026          -0.00060            0.00006     0.29425 1.00000     9     0      11
RandomExplore-10%_vs_NoExplore            recall   -0.00251          -0.00559            0.00006     0.08969 1.00000     8     0      12
RandomExplore-10%_vs_NoExplore balanced_accuracy   -0.00094          -0.00216            0.00011     0.18935 1.00000     7     0      13
RandomExplore-10%_vs_NoExplore       visible_tdr    0.00522           0.00504            0.00542     0.00000 0.00008    20     0       0
RandomExplore-10%_vs_NoExplore          fd_ratio    0.02364           0.02282            0.02452     0.00000 0.00008    20     0       0
RandomExplore-10%_vs_NoExplore             brier   -0.00007          -0.00018            0.00005     0.49801 1.00000     9     0      11
RandomExplore-10%_vs_NoExplore               ece   -0.00027          -0.00113            0.00056     0.64766 1.00000    11     0       9
RandomExplore-10%_vs_NoExplore           roc_auc   -0.00014          -0.00054            0.00027     0.43043 1.00000     8     0      12

## HMEQ significant or near-primary results

                    comparison            metric  mean_diff  bootstrap_ci_low  bootstrap_ci_high  wilcoxon_p  holm_p  wins  ties  losses
           NoExplore_vs_Oracle            recall   -0.06179          -0.06978           -0.05343     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle balanced_accuracy   -0.02577          -0.02968           -0.02175     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle       visible_tdr   -0.08072          -0.08216           -0.07932     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle          fd_ratio   -0.40636          -0.41276           -0.40022     0.00000 0.00008     0     0      20
           NoExplore_vs_Oracle             brier    0.00198           0.00132            0.00266     0.00004 0.00112    19     0       1
           NoExplore_vs_Oracle               ece    0.00990           0.00776            0.01211     0.00000 0.00008    20     0       0
           NoExplore_vs_Oracle           roc_auc    0.00171           0.00028            0.00318     0.04844 1.00000    14     0       6
 RandomExplore-2%_vs_NoExplore            recall    0.00115          -0.00627            0.00821     0.60924 1.00000     8     5       7
 RandomExplore-2%_vs_NoExplore balanced_accuracy   -0.00001          -0.00365            0.00357     0.84082 1.00000     8     0      12
 RandomExplore-2%_vs_NoExplore       visible_tdr    0.00175           0.00123            0.00226     0.00003 0.00088    19     0       1
 RandomExplore-2%_vs_NoExplore          fd_ratio    0.00879           0.00617            0.01139     0.00003 0.00088    19     0       1
 RandomExplore-2%_vs_NoExplore             brier    0.00002          -0.00045            0.00047     0.95633 1.00000    11     0       9
 RandomExplore-2%_vs_NoExplore               ece   -0.00049          -0.00259            0.00147     0.75617 1.00000     9     0      11
 RandomExplore-2%_vs_NoExplore           roc_auc   -0.00044          -0.00135            0.00045     0.31179 1.00000     7     0      13
 RandomExplore-5%_vs_NoExplore            recall    0.00322          -0.00400            0.01071     0.51966 1.00000    11     1       8
 RandomExplore-5%_vs_NoExplore balanced_accuracy    0.00060          -0.00296            0.00422     0.81236 1.00000    10     0      10
 RandomExplore-5%_vs_NoExplore       visible_tdr    0.00392           0.00347            0.00440     0.00000 0.00008    20     0       0
 RandomExplore-5%_vs_NoExplore          fd_ratio    0.01976           0.01752            0.02211     0.00000 0.00008    20     0       0
 RandomExplore-5%_vs_NoExplore             brier    0.00008          -0.00025            0.00039     0.62151 1.00000    12     0       8
 RandomExplore-5%_vs_NoExplore               ece   -0.00049          -0.00249            0.00157     0.54588 1.00000     9     0      11
 RandomExplore-5%_vs_NoExplore           roc_auc   -0.00070          -0.00155            0.00018     0.14291 1.00000     7     0      13
RandomExplore-10%_vs_NoExplore            recall    0.00629          -0.00377            0.01596     0.27724 1.00000    10     1       9
RandomExplore-10%_vs_NoExplore balanced_accuracy    0.00201          -0.00290            0.00681     0.52167 1.00000    10     0      10
RandomExplore-10%_vs_NoExplore       visible_tdr    0.00841           0.00788            0.00896     0.00000 0.00008    20     0       0
RandomExplore-10%_vs_NoExplore          fd_ratio    0.04234           0.03961            0.04505     0.00000 0.00008    20     0       0
RandomExplore-10%_vs_NoExplore             brier   -0.00025          -0.00091            0.00037     0.59582 1.00000     9     0      11
RandomExplore-10%_vs_NoExplore               ece   -0.00209          -0.00465            0.00062     0.07585 1.00000     7     0      13
RandomExplore-10%_vs_NoExplore           roc_auc   -0.00113          -0.00225            0.00008     0.06958 1.00000     7     0      13

