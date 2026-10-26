"""Document's 2x2 and quantity-relaxation diagnostics, two representative states.

This does not expand these secondary factors over the full experimental grid.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import expit
from core import *
from run_mechanisms import generate,metric,save_json

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/research_v2/focused');args=ap.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
rows=[]
for env in ['A1','A2']:
    s,yo,yc,test,yt,pt,truth=generate(env,2101)
    for method in ['OrdinaryOnly','Vopt-static-noO','Vopt-static','Vopt-noO','Vopt','Vopt-Kstar']:
        selection=select(method,s);ids=selection.selected_ids
        xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);yy=np.r_[s.visible_y,yo,yc[ids]]
        prediction=expit(test@fit_logistic(xx,yy))
        rows.append({'environment':env,'seed':2101,'method':method,'count':len(ids),'predicted_risk':selection.predicted_cost,**metric(prediction,yt,pt)})
    # Same IG score and common budget; only the count lower bound changes.
    score=ig_scores(s);k=kstar(s.cost,s.budget)
    for eps in [0.,.05,.1,.2,1.]:
        lower=int(np.ceil((1-eps)*k));ids=knapsack(s.cost,score,s.budget,lower)
        xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);yy=np.r_[s.visible_y,yo,yc[ids]]
        prediction=expit(test@fit_logistic(xx,yy))
        rows.append({'environment':env,'seed':2101,'method':'quantity_relaxation','epsilon':eps,'k_star':k,'minimum_count':lower,
                     'count':len(ids),'predicted_risk':float(s.cost[ids].sum()),'ig_sum':float(score[ids].sum()),**metric(prediction,yt,pt)})
pd.DataFrame(rows).to_csv(out/'focused_rows.csv',index=False)
save_json(out/'identity.json',{'config':json.loads((HERE/'config.json').read_text()),
                            'scope':'two representative paired mechanism states, not a full extra benchmark',
                            'source_sha256':digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'core.py',HERE/'run_mechanisms.py',Path(__file__)]})})
print('focused component/quantity diagnostics complete',len(rows))
