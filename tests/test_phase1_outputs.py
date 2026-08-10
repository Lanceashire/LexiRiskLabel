from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
T={"Oracle","NoExplore","RandomExplore-2%","RandomExplore-5%","RandomExplore-10%"}

def test_phase1_outputs_complete_and_isolated():
    for name in ["uci","hmeq"]:
        d=pd.read_csv(ROOT/"results_phase1"/f"{name}_all_rounds.csv")
        assert len(d)==1100
        assert set(d.seed)==set(range(20))
        assert set(d.treatment)==T
        assert set(d["round"])==set(range(11))
        assert d[["fixed_test_set_ok","current_round_only_ok","visible_update_ok","no_duplicate_ok","preprocessing_visible_only_ok"]].all().all()
        assert not d.treatment.str.contains("FAVE|CountOnly|Risky|Uncertainty").any()


