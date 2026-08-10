from pathlib import Path
import pandas as pd,hashlib,json,datetime
root=Path(r'D:/ecomic'); ext=root/'data/external'; pdir=ext/'ppdai'; ldir=ext/'lendingclub'; p3=root/'data/phase3'; p3.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
# PPDai: source file is the verified 55,596-row PPDai file; no credible decision-time EAD, use unit exposure.
pp=pd.read_csv(pdir/'ppdai_source.bin'); pp['exposure']=1.0; pp.to_csv(p3/'ppdai.csv',index=False)
rows=[]
for c in pp.columns:
 if c=='label': cls='TARGET';inc=False;reason='default target from source label'
 elif c=='exposure': cls='EXPOSURE_CANDIDATE';inc=False;reason='unit exposure fallback, not a model feature'
 else: cls='PRE_DECISION_FEATURE';inc=True;reason='source-provided applicant/account feature; no reliable post-outcome marker identified'
 rows.append({'column':c,'classification':cls,'included_as_feature':inc,'reason':reason})
pd.DataFrame(rows).to_csv(pdir/'PPDAI_DATA_AUDIT.csv',index=False)
(pdir/'PPDAI_DATA_AUDIT.md').write_text(f'''# PPDai data audit\n\n- Source: Figshare file 26049578, downloaded from `https://ndownloader.figshare.com/files/26049578`.\n- Local SHA-256: `{sha(pdir/'ppdai_source.bin')}`.\n- Rows: {len(pp):,}; columns: {len(pp.columns)-1} source columns plus unit exposure.\n- Target: `label` (0/1), source counts: {pp.label.value_counts().to_dict()}.\n- Exposure: no verified decision-time loan/requested amount was identified in the source schema; Phase 3 uses unit exposure (`EL=p`) and does not claim monetary loss.\n- Machine-readable field classification: `PPDAI_DATA_AUDIT.csv`.\n''',encoding='utf-8')
# LendingClub mature cohort and leakage audit.
parts=[]
for y in [2014,2015]:
 q=pd.read_csv(ldir/f'LC_Loan_sample_{y}.csv',low_memory=False); q['source_year']=y; parts.append(q)
lc=pd.concat(parts,ignore_index=True); lc=lc[lc.loan_status.isin(['Charged Off','Fully Paid'])].copy(); lc['target']=(lc.loan_status=='Charged Off').astype(int); lc=lc.drop_duplicates('id').dropna(subset=['loan_amnt']); lc['exposure']=lc.loan_amnt.astype(float)
# Strict field rules: keep only pre-decision/application fields; exclude outcome, identifiers, and payment/collection/recovery fields.
post_tokens=['total_pymnt','total_rec','recoveries','collection','last_pymnt','next_pymnt','out_prncp','last_credit_pull','settlement','hardship','payment_plan','deferral','orig_projected_additional_accrued_interest','debt_settlement']
identifier={'id','member_id','url','desc','target','exposure','loan_status','issue_d','source_year'}
rows=[]; features=[]
for c in lc.columns:
 if c in identifier: cls='OUTCOME' if c in {'loan_status','target'} else ('EXPOSURE_CANDIDATE' if c=='exposure' else ('IDENTIFIER' if c in {'id','member_id','url'} else 'AMBIGUOUS')); inc=False; reason='target/exposure/identifier or cohort metadata'
 elif any(tok in c.lower() for tok in post_tokens): cls='POST_ORIGINATION'; inc=False; reason='payment, recovery, collection, post-origination or updated-status field'
 else: cls='PRE_DECISION'; inc=True; reason='application-time field retained under frozen leakage rule'; features.append(c)
 rows.append({'column':c,'classification':cls,'included_as_feature':inc,'reason':reason})
# exclude text descriptions and object columns are allowed if they are application fields; retain all classified pre-decision.
keep=['target','exposure']+features; clean=lc[keep].copy(); clean.to_csv(p3/'lendingclub.csv',index=False)
pd.DataFrame(rows).to_csv(ldir/'LENDINGCLUB_LEAKAGE_AUDIT.csv',index=False)
status=lc.loan_status.value_counts().to_dict(); (ldir/'TARGET_MAPPING_LENDINGCLUB.csv').write_text('raw_status,mapped_class,action\nCharged Off,POSITIVE,INCLUDE\nFully Paid,NEGATIVE,INCLUDE\nCurrent,UNRESOLVED,EXCLUDE\nIn Grace Period,UNRESOLVED,EXCLUDE\nLate (16-30 days),UNRESOLVED,EXCLUDE\nLate (31-120 days),UNRESOLVED,EXCLUDE\n',encoding='utf-8')
(ldir/'LENDINGCLUB_COHORT_MANIFEST.csv').write_text(f'dataset,source_files,eligible_rows,positive_rows,negative_rows,global_sample_seed,cap_applied\nLendingClub,LC_Loan_sample_2014.csv|LC_Loan_sample_2015.csv,{len(clean)},{int(clean.target.sum())},{int((clean.target==0).sum())},20260809,NO\n',encoding='utf-8')
(ldir/'LENDINGCLUB_DATA_AUDIT.md').write_text(f'''# LendingClub data audit\n\n- Sources: public DePaul mirror listing the LendingClub 2007--2018Q4 data and 2014/2015 samples: `https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/`.\n- Mature cohort rule: include only `Charged Off` (positive) and `Fully Paid` (negative); exclude unresolved/current/late statuses.\n- Eligible rows: {len(clean):,}; positive: {int(clean.target.sum()):,}; negative: {int((clean.target==0).sum()):,}.\n- Exposure: `loan_amnt` as decision-time requested-loan proxy; `EL=p*loan_amnt`, LGD=1 proxy.\n- Leakage audit: `LENDINGCLUB_LEAKAGE_AUDIT.csv`; status mapping: `TARGET_MAPPING_LENDINGCLUB.csv`.\n''',encoding='utf-8')
(ext/'PHASE3_SOURCES.md').write_text(f'''# Phase 3 external data sources\n\n## PPDai\n\n- Figshare source file 26049578: `https://ndownloader.figshare.com/files/26049578`\n- Local file SHA-256: `{sha(pdir/'ppdai_source.bin')}`\n- Normalized rows: {len(pp):,}; target `label`; unit exposure fallback.\n- Metadata anchor: Hongliang He / DefaultData and PPDaiData collection, January 2021.\n\n## LendingClub\n\n- Public source listing and samples: `https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/`\n- 2014 SHA-256: `{sha(ldir/'LC_Loan_sample_2014.csv')}`\n- 2015 SHA-256: `{sha(ldir/'LC_Loan_sample_2015.csv')}`\n- Data dictionary SHA-256: `{sha(ldir/'LC_DataDictionary.xlsx')}`\n- Mature cohort target: Charged Off vs Fully Paid; exposure proxy `loan_amnt`.\n''',encoding='utf-8')
print('prepared',len(pp),len(clean),status)
