# DASFAA revision: numeric evidence



This version replaces negative-gap-normalized fresh-seed mechanism claims with raw improvement analysis. No model was trained; raw and historical derived files were not changed.



## Reference-gap audit

| dataset | metric | reference_cells | POSITIVE_GAP | NEGATIVE_GAP | ZERO_GAP | TINY_POSITIVE_GAP | NONFINITE | minimum_gap | maximum_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hmeq | recall | 10 | 10 | 0 | 0 | 0 | 0 | 0.03322259136212624 | 0.0903225806451613 |
| hmeq | brier | 10 | 9 | 1 | 0 | 0 | 0 | -0.0010971575975418923 | 0.004198439419269603 |
| hmeq | roc_auc | 10 | 4 | 6 | 0 | 0 | 0 | -0.006654297840950729 | 0.004376994539394441 |
| hmeq | pr_auc | 10 | 4 | 6 | 0 | 0 | 0 | -0.012597597648760894 | 0.01136367861537968 |
| hmeq | fd_ratio | 10 | 10 | 0 | 0 | 0 | 0 | 0.4038884469608347 | 0.43646375821736527 |
| lendingclub | recall | 10 | 8 | 2 | 0 | 0 | 0 | -0.006857142857142895 | 0.019011406844106404 |
| lendingclub | brier | 10 | 10 | 0 | 0 | 0 | 0 | 0.0007615685462950966 | 0.0023664608597755016 |
| lendingclub | roc_auc | 10 | 10 | 0 | 0 | 0 | 0 | 0.0005116711650338113 | 0.0035188594000776785 |
| lendingclub | pr_auc | 10 | 10 | 0 | 0 | 0 | 0 | 0.0006894373810459875 | 0.02122844446996608 |
| lendingclub | fd_ratio | 10 | 10 | 0 | 0 | 0 | 0 | 0.4834953794350334 | 0.5238649029849376 |
| ppdai | recall | 10 | 8 | 1 | 1 | 0 | 0 | -0.0005420054200542014 | 0.0061728395061727975 |
| ppdai | brier | 10 | 6 | 4 | 0 | 0 | 0 | -0.00011165440082550049 | 0.00016153603792190552 |
| ppdai | roc_auc | 10 | 5 | 5 | 0 | 0 | 0 | -0.0017377314778047603 | 0.002829216822893166 |
| ppdai | pr_auc | 10 | 6 | 4 | 0 | 0 | 0 | -0.001926864913552706 | 0.002428015237117498 |
| ppdai | fd_ratio | 10 | 10 | 0 | 0 | 0 | 0 | 0.009206548237327805 | 0.020509159186545833 |
| uci | recall | 10 | 4 | 6 | 0 | 0 | 0 | -0.01262019230769229 | 0.007854984894259898 |
| uci | brier | 10 | 10 | 0 | 0 | 0 | 0 | 9.392201900479402e-05 | 0.000955164432525607 |
| uci | roc_auc | 10 | 2 | 8 | 0 | 0 | 0 | -0.0018810425673537745 | 0.00037629430982566436 |
| uci | pr_auc | 10 | 10 | 0 | 0 | 0 | 0 | 0.0027091130185284973 | 0.013277736265025819 |
| uci | fd_ratio | 10 | 10 | 0 | 0 | 0 | 0 | 0.20615119857964892 | 0.22185048605424884 |



## Raw-gain associations (base budget and budget-centered secondary)

| dataset | rho | outcome | scope | n_cells | n_seeds | n_methods | spearman_rho | ci_low | ci_high | bootstrap_draws | valid_bootstrap_draws | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hmeq | 0.1 | recall_gain | WITHIN_BUDGET | 40 | 10 | 4 | 0.39720373160573574 | 0.013186855248257485 | 0.6599634768681827 | 10000 | 10000 | ESTIMABLE |
| hmeq | N/A | recall_gain | BUDGET_CENTERED | 200 | 10 | 4 | 0.4231433434906505 | 0.2493100705014785 | 0.5736693778305657 | 10000 | 10000 | ESTIMABLE |
| hmeq | 0.1 | brier_gain | WITHIN_BUDGET | 40 | 10 | 4 | 0.06191369606003752 | -0.175 | 0.3814485817007705 | 10000 | 10000 | ESTIMABLE |
| hmeq | N/A | brier_gain | BUDGET_CENTERED | 200 | 10 | 4 | -0.040928523213080326 | -0.21360781264081385 | 0.15434379572893456 | 10000 | 10000 | ESTIMABLE |
| lendingclub | 0.1 | recall_gain | WITHIN_BUDGET | 40 | 10 | 4 | -0.0034714078309711098 | -0.32647592160524724 | 0.42262087488902794 | 10000 | 10000 | ESTIMABLE |
| lendingclub | N/A | recall_gain | BUDGET_CENTERED | 200 | 10 | 4 | -0.14361573666524147 | -0.3639850695276952 | 0.1306302995660609 | 10000 | 10000 | ESTIMABLE |
| lendingclub | 0.1 | brier_gain | WITHIN_BUDGET | 40 | 10 | 4 | -0.017823639774859287 | -0.3024464526066954 | 0.3881184321135156 | 10000 | 10000 | ESTIMABLE |
| lendingclub | N/A | brier_gain | BUDGET_CENTERED | 200 | 10 | 4 | 0.017684698749232005 | -0.18916275379366382 | 0.2988194594049532 | 10000 | 10000 | ESTIMABLE |
| ppdai | 0.1 | recall_gain | WITHIN_BUDGET | 40 | 10 | 4 | 0.07063022728018857 | -0.3195274464407871 | 0.48089644233379614 | 10000 | 10000 | ESTIMABLE |
| ppdai | N/A | recall_gain | BUDGET_CENTERED | 200 | 10 | 4 | 0.18317864562859298 | -0.2654340603830039 | 0.608273530943822 | 10000 | 10000 | ESTIMABLE |
| ppdai | 0.1 | brier_gain | WITHIN_BUDGET | 40 | 10 | 4 | -0.008262910798122065 | -0.3486508632209097 | 0.3922831833620825 | 10000 | 10000 | ESTIMABLE |
| ppdai | N/A | brier_gain | BUDGET_CENTERED | 200 | 10 | 4 | 0.16051430863273117 | -0.1754124349025626 | 0.5037908295837439 | 10000 | 10000 | ESTIMABLE |
| uci | 0.1 | recall_gain | WITHIN_BUDGET | 40 | 10 | 4 | 0.07699892022196213 | -0.3409204371548503 | 0.45920777988614797 | 10000 | 10000 | ESTIMABLE |
| uci | N/A | recall_gain | BUDGET_CENTERED | 200 | 10 | 4 | -0.12581062290046513 | -0.44102378564719 | 0.2612177086243933 | 10000 | 10000 | ESTIMABLE |
| uci | 0.1 | brier_gain | WITHIN_BUDGET | 40 | 10 | 4 | 0.09596244131455399 | -0.34222716976282147 | 0.5456907002149627 | 10000 | 10000 | ESTIMABLE |
| uci | N/A | brier_gain | BUDGET_CENTERED | 200 | 10 | 4 | 0.08069004312769548 | -0.22874949480089476 | 0.44073217008654786 | 10000 | 10000 | ESTIMABLE |



## Primary paired comparisons (left minus right; Brier negative is better)

| dataset | left | right | metric | n_paired_seeds | left_mean | right_mean | mean_difference | ci_low | ci_high | rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hmeq | LRBE-FAVE | CountOnly-MinRisk | cumulative_feedback_count | 10 | 157.6 | 156.5 | 1.1 | -0.9 | 3.0 | 0.1 |
| hmeq | LRBE-FAVE | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 692812.8632700443 | 660207.8279393911 | 32605.03533065319 | 22925.21690221131 | 43035.32137353718 | 0.1 |
| hmeq | LRBE-FAVE | CountOnly-MinRisk | recall | 10 | 0.6213098126782066 | 0.6219850602372446 | -0.0006752475590379615 | -0.00828493183664869 | 0.00916284469645945 | 0.1 |
| hmeq | LRBE-FAVE | CountOnly-MinRisk | brier | 10 | 0.07732421383261676 | 0.07714331224560733 | 0.00018090158700942855 | -0.00011378617957234125 | 0.000555758513510217 | 0.1 |
| hmeq | LRBE-FAVE | QualityFirst-FAVE | cumulative_feedback_count | 10 | 157.6 | 90.5 | 67.1 | 62.6 | 71.4 | 0.1 |
| hmeq | LRBE-FAVE | QualityFirst-FAVE | cumulative_predicted_risk | 10 | 692812.8632700443 | 679547.6402801275 | 13265.222989916801 | 5393.4420397132635 | 22235.538200885057 | 0.1 |
| hmeq | LRBE-FAVE | QualityFirst-FAVE | recall | 10 | 0.6213098126782066 | 0.603854303532825 | 0.017455509145381888 | 0.0040558548985601545 | 0.029716916218262078 | 0.1 |
| hmeq | LRBE-FAVE | QualityFirst-FAVE | brier | 10 | 0.07732421383261676 | 0.07762108594179147 | -0.00029687210917471034 | -0.0010699234902858388 | 0.0004846164770424396 | 0.1 |
| hmeq | RiskBounded-Random | CountOnly-MinRisk | cumulative_feedback_count | 10 | 69.5 | 156.5 | -87.0 | -91.5 | -82.5 | 0.1 |
| hmeq | RiskBounded-Random | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 669232.9811537266 | 660207.8279393911 | 9025.153214335442 | -4216.38949881494 | 23349.804284453392 | 0.1 |
| hmeq | RiskBounded-Random | CountOnly-MinRisk | recall | 10 | 0.5952251585448256 | 0.6219850602372446 | -0.026759901692419208 | -0.03954636308722685 | -0.014038137104056173 | 0.1 |
| hmeq | RiskBounded-Random | CountOnly-MinRisk | brier | 10 | 0.07760245427489279 | 0.07714331224560733 | 0.0004591420292854503 | -0.00015335870906706378 | 0.001127669140696546 | 0.1 |
| hmeq | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_feedback_count | 10 | 157.8 | 156.5 | 1.3 | 0.1 | 2.4 | 0.1 |
| hmeq | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 681067.7044427395 | 660207.8279393911 | 20859.87650334835 | 11452.714189529419 | 30748.453647002574 | 0.1 |
| hmeq | RiskBounded-Epsilon | CountOnly-MinRisk | recall | 10 | 0.6227731478446096 | 0.6219850602372446 | 0.0007880876073648513 | -0.006046401504620804 | 0.008184011465991408 | 0.1 |
| hmeq | RiskBounded-Epsilon | CountOnly-MinRisk | brier | 10 | 0.07715755850076672 | 0.07714331224560733 | 1.4246255159390541e-05 | -0.0004383277148008311 | 0.0004179723002016505 | 0.1 |
| lendingclub | LRBE-FAVE | CountOnly-MinRisk | cumulative_feedback_count | 10 | 535.1 | 535.2 | -0.1 | -2.5 | 2.6 | 0.1 |
| lendingclub | LRBE-FAVE | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 2453910.516668707 | 2422361.7091690004 | 31548.807499706745 | 26822.867989376093 | 35982.23959299181 | 0.1 |
| lendingclub | LRBE-FAVE | CountOnly-MinRisk | recall | 10 | 0.7294925537366583 | 0.7294646423185291 | 2.7911418129245025e-05 | -0.003925653068143667 | 0.004183268545378519 | 0.1 |
| lendingclub | LRBE-FAVE | CountOnly-MinRisk | brier | 10 | 0.07331225648522374 | 0.07315418794751163 | 0.00015806853771210827 | -2.6911944150918055e-05 | 0.00036987736821175873 | 0.1 |
| lendingclub | LRBE-FAVE | QualityFirst-FAVE | cumulative_feedback_count | 10 | 535.1 | 325.5 | 209.6 | 202.5 | 216.1 | 0.1 |
| lendingclub | LRBE-FAVE | QualityFirst-FAVE | cumulative_predicted_risk | 10 | 2453910.516668707 | 2475582.429045737 | -21671.912377029705 | -35964.834736716 | -6628.789552323577 | 0.1 |
| lendingclub | LRBE-FAVE | QualityFirst-FAVE | recall | 10 | 0.7294925537366583 | 0.7246766470033787 | 0.004815906733279585 | -0.0033745763465433025 | 0.01340440629771931 | 0.1 |
| lendingclub | LRBE-FAVE | QualityFirst-FAVE | brier | 10 | 0.07331225648522374 | 0.07343303933739659 | -0.00012078285217283907 | -0.0003161485120653872 | 7.220443338155909e-05 | 0.1 |
| lendingclub | RiskBounded-Random | CountOnly-MinRisk | cumulative_feedback_count | 10 | 217.6 | 535.2 | -317.6 | -327.0 | -308.0 | 0.1 |
| lendingclub | RiskBounded-Random | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 2458968.6041012406 | 2422361.7091690004 | 36606.89493224025 | 7542.85217389469 | 61857.201672643314 | 0.1 |
| lendingclub | RiskBounded-Random | CountOnly-MinRisk | recall | 10 | 0.7266284854504319 | 0.7294646423185291 | -0.002836156868097317 | -0.009517805285972377 | 0.004172177990095968 | 0.1 |
| lendingclub | RiskBounded-Random | CountOnly-MinRisk | brier | 10 | 0.07345747947692866 | 0.07315418794751163 | 0.00030329152941702684 | -8.805952966214448e-05 | 0.0006963955797254749 | 0.1 |
| lendingclub | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_feedback_count | 10 | 533.7 | 535.2 | -1.5 | -4.2 | 1.4 | 0.1 |
| lendingclub | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 2447085.3645798564 | 2422361.7091690004 | 24723.65541085601 | 13689.337572462902 | 36067.73834038908 | 0.1 |
| lendingclub | RiskBounded-Epsilon | CountOnly-MinRisk | recall | 10 | 0.7307831121697989 | 0.7294646423185291 | 0.0013184698512697436 | -0.0010626333394376328 | 0.0037617402539695277 | 0.1 |
| lendingclub | RiskBounded-Epsilon | CountOnly-MinRisk | brier | 10 | 0.07298230975866313 | 0.07315418794751163 | -0.00017187818884849133 | -0.0004244340583681876 | 5.532972514627711e-05 | 0.1 |
| ppdai | LRBE-FAVE | CountOnly-MinRisk | cumulative_feedback_count | 10 | 13.3 | 13.3 | 0.0 | -1.0 | 1.1 | 0.1 |
| ppdai | LRBE-FAVE | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 7.416245841979981 | 6.734591180086136 | 0.6816546618938448 | 0.14172136187553405 | 1.278558105677366 | 0.1 |
| ppdai | LRBE-FAVE | CountOnly-MinRisk | recall | 10 | 0.01932015986493746 | 0.01953841902890037 | -0.0002182591639629104 | -0.0027600567150234986 | 0.0023273938990919394 | 0.1 |
| ppdai | LRBE-FAVE | CountOnly-MinRisk | brier | 10 | 0.1049496196210384 | 0.10496954321861263 | -1.9923597574236783e-05 | -6.193710491062199e-05 | 1.9280333071952504e-05 | 0.1 |
| ppdai | LRBE-FAVE | QualityFirst-FAVE | cumulative_feedback_count | 10 | 13.3 | 11.3 | 2.0 | 0.9 | 3.1 | 0.1 |
| ppdai | LRBE-FAVE | QualityFirst-FAVE | cumulative_predicted_risk | 10 | 7.416245841979981 | 6.330721294879913 | 1.0855245471000672 | 0.5630144460499287 | 1.6800156135857105 | 0.1 |
| ppdai | LRBE-FAVE | QualityFirst-FAVE | recall | 10 | 0.01932015986493746 | 0.018815610632115797 | 0.0005045492328216598 | -0.001123761830685701 | 0.0020047908759265286 | 0.1 |
| ppdai | LRBE-FAVE | QualityFirst-FAVE | brier | 10 | 0.1049496196210384 | 0.1049817010760307 | -3.208145499231929e-05 | -9.065486490731178e-05 | 2.609694376583535e-05 | 0.1 |
| ppdai | RiskBounded-Random | CountOnly-MinRisk | cumulative_feedback_count | 10 | 13.0 | 13.3 | -0.3 | -1.1 | 0.6 | 0.1 |
| ppdai | RiskBounded-Random | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 7.376957058906555 | 6.734591180086136 | 0.6423658788204195 | 0.2989397217333318 | 0.9991193579137323 | 0.1 |
| ppdai | RiskBounded-Random | CountOnly-MinRisk | recall | 10 | 0.01875356689531726 | 0.01953841902890037 | -0.0007848521335831103 | -0.002001213027461208 | 0.0005074891257344257 | 0.1 |
| ppdai | RiskBounded-Random | CountOnly-MinRisk | brier | 10 | 0.10497562661767001 | 0.10496954321861263 | 6.083399057393857e-06 | -4.5729875564564545e-05 | 6.209526211023634e-05 | 0.1 |
| ppdai | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_feedback_count | 10 | 13.6 | 13.3 | 0.3 | -0.2 | 0.8 | 0.1 |
| ppdai | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 6.921546453237534 | 6.734591180086136 | 0.18695527315139776 | -0.0643938963115217 | 0.4137898625433423 | 0.1 |
| ppdai | RiskBounded-Epsilon | CountOnly-MinRisk | recall | 10 | 0.01966766036825523 | 0.01953841902890037 | 0.00012924133935485974 | -0.0015483421373314703 | 0.0015231003062256808 | 0.1 |
| ppdai | RiskBounded-Epsilon | CountOnly-MinRisk | brier | 10 | 0.10492414385080333 | 0.10496954321861263 | -4.539936780928455e-05 | -0.00012149173766376082 | 3.0279159545933132e-05 | 0.1 |
| uci | LRBE-FAVE | CountOnly-MinRisk | cumulative_feedback_count | 10 | 273.5 | 273.8 | -0.3 | -2.0 | 1.5 | 0.1 |
| uci | LRBE-FAVE | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 147.7366293311119 | 145.58051711916923 | 2.1561122119426726 | 1.3517495444417 | 2.997378922700882 | 0.1 |
| uci | LRBE-FAVE | CountOnly-MinRisk | recall | 10 | 0.3634935630028893 | 0.3638927412957994 | -0.00039917829291016285 | -0.0033054450858647833 | 0.0028932225815693846 | 0.1 |
| uci | LRBE-FAVE | CountOnly-MinRisk | brier | 10 | 0.13405952304601665 | 0.13423731327056881 | -0.00017779022455215453 | -0.00028021872043609895 | -8.193906396626244e-05 | 0.1 |
| uci | LRBE-FAVE | QualityFirst-FAVE | cumulative_feedback_count | 10 | 273.5 | 217.1 | 56.4 | 54.7 | 58.2 | 0.1 |
| uci | LRBE-FAVE | QualityFirst-FAVE | cumulative_predicted_risk | 10 | 147.7366293311119 | 143.6780512571335 | 4.058578073978424 | 3.162531076073647 | 4.926891327947378 | 0.1 |
| uci | LRBE-FAVE | QualityFirst-FAVE | recall | 10 | 0.3634935630028893 | 0.36390803177059555 | -0.00041446876770630524 | -0.0048125230898599415 | 0.004106185996876779 | 0.1 |
| uci | LRBE-FAVE | QualityFirst-FAVE | brier | 10 | 0.13405952304601665 | 0.13401575833559037 | 4.3764710426330566e-05 | -7.133696228266038e-05 | 0.00015599299222233417 | 0.1 |
| uci | RiskBounded-Random | CountOnly-MinRisk | cumulative_feedback_count | 10 | 214.3 | 273.8 | -59.5 | -62.1 | -56.8 | 0.1 |
| uci | RiskBounded-Random | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 144.96208971738815 | 145.58051711916923 | -0.618427401781085 | -2.0752787643671042 | 0.8591770960390539 | 0.1 |
| uci | RiskBounded-Random | CountOnly-MinRisk | recall | 10 | 0.36346596407842896 | 0.3638927412957994 | -0.0004267772173704498 | -0.003839216105746428 | 0.0032422302231753077 | 0.1 |
| uci | RiskBounded-Random | CountOnly-MinRisk | brier | 10 | 0.1341850250959396 | 0.13423731327056881 | -5.2288174629222525e-05 | -0.0002755286544561606 | 0.00015362061560153387 | 0.1 |
| uci | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_feedback_count | 10 | 271.1 | 273.8 | -2.7 | -3.9 | -1.3 | 0.1 |
| uci | RiskBounded-Epsilon | CountOnly-MinRisk | cumulative_predicted_risk | 10 | 144.6566098690033 | 145.58051711916923 | -0.9239072501659393 | -1.8261265344917774 | 0.018637429177761056 | 0.1 |
| uci | RiskBounded-Epsilon | CountOnly-MinRisk | recall | 10 | 0.3622407857605193 | 0.3638927412957994 | -0.0016519555352800363 | -0.004450336032001334 | 0.0011476381951218964 | 0.1 |
| uci | RiskBounded-Epsilon | CountOnly-MinRisk | brier | 10 | 0.13418546766042705 | 0.13423731327056881 | -5.1845610141748597e-05 | -0.00016307413578035363 | 4.989881068469765e-05 | 0.1 |



## Interpretation limits

All comparisons are policy-level under paired exogenous initial conditions, not shared later candidates/numeric budgets. Similar mean counts and intervals crossing zero are not matching interventions or equivalence tests. Associations do not establish independence or causal composition value. Frozen Phase 4 associations have not been recomputed; do not present them as updated evidence.



Positive-gap-only recovery is supplementary and unavailable for negative, zero, tiny-positive (<=1e-10), or nonfinite gaps. Numerical tolerance is inherited, not tuned. Gains remain primary and retain negative-gap cells. FDRatio gain is change in a composition ratio, not a calibrated measure of representativeness; a larger ratio need not be closer to 1.



Runtime values are descriptive historical end-to-end wall times affected by concurrent execution. They cannot identify algorithmic overhead. See analysis_settings.json and descriptive_episode_runtime.csv.



Reproduce: D:/envs_dirs/no_1/python.exe experiments/ccfb_extension/analyze_dasfaa_revision.py
