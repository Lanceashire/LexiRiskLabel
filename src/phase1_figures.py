from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results_phase1"; FIG=RES/"figures"; FIG.mkdir(exist_ok=True)
T=["Oracle","NoExplore","RandomExplore-2%","RandomExplore-5%","RandomExplore-10%"]
COL={"Oracle":(30,90,180),"NoExplore":(210,50,50),"RandomExplore-2%":(40,150,70),"RandomExplore-5%":(230,140,20),"RandomExplore-10%":(140,60,160)}
for name in ["uci","hmeq"]:
    d=pd.read_csv(RES/f"{name}_all_rounds.csv")
    for metric in ["visible_tdr","fd_ratio","recall","balanced_accuracy","roc_auc","brier","ece"]:
        series={}
        for t in T:
            g=d[d.treatment==t].groupby("round")[metric].agg(["mean","std","count"]).reindex(range(11))
            series[t]=g
        vals=np.concatenate([g["mean"].to_numpy() for g in series.values()])
        vals=vals[np.isfinite(vals)]
        lo=float(vals.min()); hi=float(vals.max()); pad=max((hi-lo)*.12,1e-3); lo-=pad; hi+=pad
        W,H=1000,600; ml,mr,mt,mb=90,30,45,75; pw,ph=W-ml-mr,H-mt-mb
        im=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(im)
        dr.line((ml,mt,ml,H-mb),fill="black",width=2); dr.line((ml,H-mb,W-mr,H-mb),fill="black",width=2)
        for k in range(6):
            y=mt+ph*k/5; val=hi-(hi-lo)*k/5
            dr.line((ml-5,y,ml,y),fill="black",width=1); dr.text((5,y-8),f"{val:.3f}",fill="black")
        for r in range(11):
            x=ml+pw*r/10; dr.line((x,H-mb,x,H-mb+5),fill="black",width=1); dr.text((x-5,H-mb+8),str(r),fill="black")
        dr.text((W//2-100,10),f"{name.upper()} {metric} trajectory",fill="black")
        for t in T:
            g=series[t]; xs=[]; ys=[]; upper=[]; lower=[]
            for r in range(11):
                m=float(g.loc[r,"mean"]); sd=float(g.loc[r,"std"]); n=float(g.loc[r,"count"])
                if not np.isfinite(m): continue
                x=ml+pw*r/10; y=mt+ph*(hi-m)/(hi-lo); ci=1.96*sd/np.sqrt(n) if np.isfinite(sd) and n>0 else 0
                xs.append(x); ys.append(y); upper.append(mt+ph*(hi-(m+ci))/(hi-lo)); lower.append(mt+ph*(hi-(m-ci))/(hi-lo))
            if len(xs)>1:
                poly=list(zip(xs,upper))+list(zip(xs[::-1],lower[::-1])); dr.polygon(poly,fill=tuple(min(255,int(c*.88+255*.12)) for c in COL[t]))
                dr.line(list(zip(xs,ys)),fill=COL[t],width=3)
        ly=mt
        for t in T:
            dr.line((W-270,ly,W-245,ly),fill=COL[t],width=4); dr.text((W-235,ly-8),t,fill="black"); ly+=24
        im.save(FIG/f"{name}_{metric}_trajectory.png")
print("figures generated")


