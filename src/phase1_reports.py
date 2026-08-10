from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]; R=root/"results_phase1"
lines=["# Phase 1 - Selective-label Feedback Mechanism Validation","","## Status","","PHASE1_STATUS = PASS_WITH_MIXED_MECHANISMS","","Protocol: PASS; UCI: COMPLETE (20/20 seeds); HMEQ: COMPLETE (20/20 seeds).","Phase 1 is complete. Per protocol, do not enter Phase 2 automatically.","","## Treatments","","Oracle, NoExplore, RandomExplore-2%, RandomExplore-5%, RandomExplore-10%. FAVE and risk-budget policies were excluded.",""]
for n in ["uci","hmeq"]:
    d=pd.read_csv(R/f"{n}_all_rounds.csv"); f=d[d["round"]==10]
    tab=f.groupby("treatment")[["recall","balanced_accuracy","visible_tdr","fd_ratio","brier","ece","roc_auc","pr_auc"]].mean().round(4)
    lines += [f"## {n.upper()} final round (mean over 20 seeds)","",tab.to_string(),""]

lines += ["## Hypothesis decisions","","- H1 Selective-label feedback distorts visible TDR/FDRatio: SUPPORTED on both datasets. NoExplore has lower Visible TDR and FDRatio than Oracle; paired Holm-adjusted tests support this.","- H2 Default-class recall/BA may degrade versus Oracle: SUPPORTED for HMEQ (Recall and Balanced Accuracy lower than Oracle), NOT SUPPORTED for UCI (Recall/BA do not show the same degradation).","- H3 Accuracy/AUC may remain stable while recall/TDR degrade: SUPPORTED as a mechanism pattern. ROC-AUC remains close to Oracle while feedback health is distorted; the direction of Recall is dataset-dependent.","- H4 Calibration may deteriorate without ranking collapse: MIXED / DATASET-DEPENDENT. HMEQ shows worse Brier/ECE under NoExplore; UCI shows a smaller and mixed calibration change.","- H5 Simple random exploration may recover part of the Oracle gap: SUPPORTED for feedback health (TDR/FDRatio move toward Oracle), but decision/calibration recovery is weak or inconclusive after Holm correction.","","## Interpretation","","Selective labels first change which defaults become visible, not necessarily ranking discrimination. HMEQ exhibits a clear Recall/BA and calibration penalty, while UCI mainly shows TDR/FDRatio distortion with stable AUC and no matching Recall collapse. Random current-round exploration recovers part of the feedback-health gap, but its recovery of decision and calibration endpoints is dataset-dependent.","","## Post-hoc Findings","","- Dual evaluation is post-hoc and is not assigned a new H-number. The external fixed-test evaluator exactly reproduces the frozen all-rounds results; the internal visible-only evaluator produces systematic Oracle--NoExplore distortion in both datasets. See PHASE1_ADDENDUM_DUAL_EVALUATION.md and dual_evaluation_statistics.csv.","","## Stop condition","","This report is the terminal Phase 1 output. No FAVE comparison, risk-budget optimization, new acquisition policy, PPDai/LendingClub expansion, or Phase 2 execution was performed."]
(R/"PHASE1_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

stat=["# Phase 1 statistical report","","All comparisons are paired by seed (20 seeds per dataset). Wilcoxon signed-rank tests use Holm correction within each dataset across the reported comparisons and metrics. Bootstrap confidence intervals use 10,000 fixed-seed resamples of paired seed differences.",""]
for n in ["uci","hmeq"]:
    t=pd.read_csv(R/n/"paired_tests.csv")
    stat += [f"## {n.upper()} significant or near-primary results","",t[t.metric.isin(["recall","balanced_accuracy","visible_tdr","fd_ratio","brier","ece","roc_auc"])][["comparison","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","holm_p","wins","ties","losses"]].round(5).to_string(index=False),""]
(R/"PHASE1_STATISTICAL_REPORT.md").write_text("\n".join(stat)+"\n",encoding="utf-8")

mech=["# Phase 1 mechanism report","","The mechanism is selective-label feedback: rejected applicants are not observed unless they are in the Oracle treatment or selected by current-round random exploration.","","## Mechanistic conclusions","","1. Visible TDR and FDRatio are the most consistent distortion signals across UCI and HMEQ.","2. ROC-AUC is comparatively stable, so aggregate ranking metrics can conceal feedback degradation.","3. HMEQ shows a stronger default-class and calibration penalty than UCI.","4. Random exploration improves observed feedback health monotonically with rate, but it does not guarantee recovery of Recall, BA, or calibration.","5. The effect is dataset-dependent; no forced cross-dataset unification is made.","","See PHASE1_INTEGRITY_AUDIT.md, trajectory_summary.csv, threshold_crossing_summary.csv, and reliability_bins.csv for audit and mechanism evidence."]
(R/"PHASE1_MECHANISM_REPORT.md").write_text("\n".join(mech)+"\n",encoding="utf-8")
print("reports generated")



