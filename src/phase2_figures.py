from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]; S=ROOT/'results_phase2/summary'; F=ROOT/'results_phase2/figures'; F.mkdir(parents=True,exist_ok=True)
font='C:/Windows/Fonts/arial.ttf'; bold='C:/Windows/Fonts/arialbd.ttf'
def ft(n,b=False):
    try:return ImageFont.truetype(bold if b else font,n)
    except:return ImageFont.load_default()
COL={'CountOnly':'#666666','CountOnly-MinRisk':'#2166ac','LeastRisky':'#1b9e77','MostRisky':'#d95f02','Uncertainty':'#7570b3','RiskTilted-Uncertainty':'#e7298a','FAVE-v2.1':'#000000','NoExplore':'#b2182b'}
def lineplot(ds,metric,title,ylabel,filename):
    d=pd.read_csv(S/'final_summary.csv'); d=d[d.dataset==ds]; W,H=1250,780; im=Image.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(im); L,T,R,B=105,90,1170,650
    dr.text((L,25),title,font=ft(27,True),fill='black'); dr.text((25,340),ylabel,font=ft(17),fill='black')
    vals=d[metric+'_mean'].replace([np.inf,-np.inf],np.nan).dropna(); lo=float(vals.min()) if len(vals) else 0; hi=float(vals.max()) if len(vals) else 1; pad=max((hi-lo)*.1,0.02); lo-=pad; hi+=pad
    for j in range(6):
      y=T+(B-T)*j/5; v=hi-(hi-lo)*j/5; dr.line((L,y,R,y),fill='#dddddd'); dr.text((50,y-8),f'{v:.3f}',font=ft(14),fill='#555')
    for xval in [.025,.05,.10,.20]:
      x=L+(R-L)*([.025,.05,.10,.20].index(xval))/3; dr.line((x,T,x,B),fill='#eeeeee'); dr.text((x-18,B+12),f'{xval*100:.1f}%',font=ft(14),fill='#555')
    for pol in d.policy.unique():
      q=d[d.policy==pol].set_index('rho').reindex([.025,.05,.10,.20]); pts=[]
      for xval,v in zip([.025,.05,.10,.20],q[metric+'_mean']):
        if pd.isna(v): continue
        x=L+(R-L)*([.025,.05,.10,.20].index(xval))/3; y=B-(B-T)*(float(v)-lo)/(hi-lo); pts.append((x,y))
      for a,b in zip(pts[:-1],pts[1:]): dr.line((*a,*b),fill=COL.get(pol,'#444'),width=3)
      for x,y in pts: dr.ellipse((x-4,y-4,x+4,y+4),fill=COL.get(pol,'#444'))
    for i,pol in enumerate(d.policy.unique()):
      x=810+(i%2)*180; y=30+(i//2)*24; dr.line((x,y+9,x+25,y+9),fill=COL.get(pol,'#444'),width=3); dr.text((x+32,y),pol,font=ft(13),fill='black')
    im.save(F/filename)
for ds in ['uci','hmeq']:
  for metric,ylabel in [('recall','Final recall'),('fd_ratio','Final FDRatio'),('visible_tdr','Final visible TDR'),('brier','Final Brier'),('ece','Final ECE'),('cumulative_explored_labels','Cumulative explored labels')]: lineplot(ds,metric,f'{ds.upper()} Phase 2: {ylabel}',ylabel,f'{ds}_{metric}_vs_budget.png')

def pareto(ds,ycol,xcol,title,filename):
 d=pd.read_csv(S/'pareto_points.csv'); d=d[d.dataset==ds]; W,H=1100,760; im=Image.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(im);L,T,R,B=100,90,1030,640
 dr.text((L,25),title,font=ft(26,True),fill='black'); xs=d[xcol].to_numpy(float); ys=d[ycol].to_numpy(float); xmin,xmax=float(np.nanmin(xs)),float(np.nanmax(xs)); ymin,ymax=float(np.nanmin(ys)),float(np.nanmax(ys)); xp=max((xmax-xmin)*.1,1e-6); yp=max((ymax-ymin)*.1,1e-6); xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp
 for j in range(6):
  y=T+(B-T)*j/5; dr.line((L,y,R,y),fill='#dddddd'); dr.text((45,y-8),f'{ymax-(ymax-ymin)*j/5:.3f}',font=ft(13),fill='#555')
 for row in d.itertuples():
  x=L+(R-L)*(row.__getattribute__(xcol)-xmin)/(xmax-xmin); y=B-(B-T)*(row.__getattribute__(ycol)-ymin)/(ymax-ymin); col=COL.get(row.policy,'#444'); dr.ellipse((x-6,y-6,x+6,y+6),fill=col); dr.text((x+8,y-8),f'{row.policy[:8]} {row.rho:.3g}',font=ft(11),fill=col)
 dr.text((L,B+35),xcol,font=ft(16),fill='black'); im.save(F/filename)
for ds in ['uci','hmeq']:
  pareto(ds,'recall_gain','x_cumulative_expected_risk',f'{ds.upper()} Pareto: recall gain vs expected risk','%s_pareto_recall.png'%ds); pareto(ds,'fdratio_gain','x_cumulative_expected_risk',f'{ds.upper()} Pareto: FDRatio gain vs expected risk','%s_pareto_fdratio.png'%ds); pareto(ds,'brier_improvement','x_cumulative_expected_risk',f'{ds.upper()} Pareto: Brier improvement vs expected risk','%s_pareto_brier.png'%ds)
print('phase2 figures complete')

