"""Reconstruct public inputs without replacing frozen study files."""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);src=a.source
    h=pd.read_csv(src/'hmeq_sas.csv');assert len(h)==5960 and set(h.BAD.unique())=={0,1}
    h.to_csv(a.output/'hmeq.csv',index=False)
    u=pd.read_excel(src/'uci_source/default of credit card clients.xls',header=1).rename(columns={'default payment next month':'default'})
    assert len(u)==30000;u.to_csv(a.output/'uci.csv',index=False)
    parts=[]
    for year in [2014,2015]:
        x=pd.read_csv(src/f'LC_Loan_sample_{year}.csv',low_memory=False);x['source_year']=year;parts.append(x)
    lc=pd.concat(parts,ignore_index=True);lc=lc[lc.loan_status.isin(['Charged Off','Fully Paid'])].copy()
    lc['target']=(lc.loan_status=='Charged Off').astype(int);lc=lc.drop_duplicates('id').dropna(subset=['loan_amnt']);lc['exposure']=lc.loan_amnt.astype(float)
    tokens=['total_pymnt','total_rec','recoveries','collection','last_pymnt','next_pymnt','out_prncp','last_credit_pull','settlement','hardship','payment_plan','deferral','orig_projected_additional_accrued_interest','debt_settlement']
    excluded={'id','member_id','url','desc','target','exposure','loan_status','issue_d','source_year'}
    features=[c for c in lc if c not in excluded and not any(t in c.lower() for t in tokens)]
    clean=lc[['target','exposure']+features];assert len(clean)==16566 and clean.target.sum()==3321
    clean.to_csv(a.output/'lendingclub.csv',index=False)
    # Identity retained separately for applicant-disjoint evaluation.
    lc[['id','issue_d','source_year']].to_csv(a.output/'lendingclub_ids.csv',index=False)
    root=Path(__file__).resolve().parents[2]
    registry=json.loads((root/'experiments/information_gain/input_registry.json').read_text())
    provenance=[]
    for name,frame in [('hmeq',h),('uci',u),('lendingclub',clean)]:
        p=a.output/f'{name}.csv';actual=sha(p)
        provenance.append({'dataset':name,'rows':len(frame),'sha256':actual,'registered_sha256':registry[name]['sha256'],
                           'exact_byte_parity':actual==registry[name]['sha256'],
                           'role':'new v2 source reconstruction; do not merge with frozen runs unless exact parity is established'})
    (a.output/'provenance.json').write_text(json.dumps({'inputs':provenance,'sources':[
        {'url':'https://github.com/sassoftware/sas-viya-dmml-pipelines/blob/master/data/hmeq.csv','local':str(src/'hmeq_sas.csv'),'sha256':sha(src/'hmeq_sas.csv')},
        {'url':'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients','local':str(src/'uci.zip'),'sha256':sha(src/'uci.zip')},
        *[{'url':f'https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/LC_Loan_sample_{year}.csv','local':str(src/f'LC_Loan_sample_{year}.csv'),'sha256':sha(src/f'LC_Loan_sample_{year}.csv')} for year in [2014,2015]]]},indent=2))
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
