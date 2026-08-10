from pathlib import Path
import pandas as pd, numpy as np
from PIL import Image, ImageDraw, ImageFont

RES=Path(__file__).resolve().parents[1]/'results_phase1'; FIG=RES/'figures'; FIG.mkdir(exist_ok=True)
df=pd.read_csv(RES/'dual_evaluation_all_rounds.csv')
font='/Windows/Fonts/arial.ttf'; bold='/Windows/Fonts/arialbd.ttf'
def f(sz,b=False):
    try:return ImageFont.truetype(bold if b else font,sz)
    except:return ImageFont.load_default()
def plot(name,metric,title,ylabel):
    W,H=1200,760; im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
    L,T,R,B=105,90,1140,650
    d.text((L,25),title,font=f(28,True),fill='black'); d.text((25,330),ylabel,font=f(18),fill='black')
    for j in range(6):
        y=T+(B-T)*j/5; val=1-j/5
        d.line((L,y,R,y),fill='#dddddd',width=1); d.text((55,y-9),f'{val:.1f}',font=f(15),fill='#555555')
    for t in range(11):
        x=L+(R-L)*t/10; d.line((x,T,x,B),fill='#eeeeee',width=1); d.text((x-8,B+12),str(t),font=f(15),fill='#555555')
    colors={'Oracle':'#2166ac','NoExplore':'#b2182b'}; styles={'external_fixed':'-','internal_visible':'--'}
    for scope in ['external_fixed','internal_visible']:
      for tr in ['Oracle','NoExplore']:
        s=df[(df.dataset==name)&(df.evaluation_scope==scope)&(df.treatment==tr)].groupby('round')[metric].mean().reindex(range(11)); vals=s.to_numpy(float)
        pts=[]
        for t,v in enumerate(vals): pts.append((L+(R-L)*t/10,B-(B-T)*np.clip(v,0,1)))
        for a,b in zip(pts[:-1],pts[1:]): d.line((*a,*b),fill=colors[tr],width=3)
        for x,y in pts:d.ellipse((x-4,y-4,x+4,y+4),fill=colors[tr])
    legend=[('Oracle / external','#2166ac','solid'),('NoExplore / external','#b2182b','solid'),('Oracle / internal','#2166ac','dash'),('NoExplore / internal','#b2182b','dash')]
    x0,y0=760,35
    for i,(lab,col,sty) in enumerate(legend):
      yy=y0+i*24; d.line((x0,yy+9,x0+35,yy+9),fill=col,width=3); d.text((x0+45,yy),lab,font=f(15),fill='black')
    im.save(FIG/name)
for ds in ['uci','hmeq']:
  plot(ds+'_dual_recall.png','recall',ds.upper()+' dual evaluation: recall','Recall')
  plot(ds+'_dual_roc_auc.png','roc_auc',ds.upper()+' dual evaluation: ROC-AUC','ROC-AUC')
print('dual figures complete')

