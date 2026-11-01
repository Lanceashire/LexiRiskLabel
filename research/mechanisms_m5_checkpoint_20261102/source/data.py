"""Fixed source schema and covariate-only group/split construction."""
from common import *
import io,zipfile
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
LC_COLS=['loan_amnt','term','int_rate','grade','emp_length','home_ownership','annual_inc','purpose','dti','delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status']
SG_COLS=['laufkont','laufzeit','moral','verw','hoehe','sparkont','beszeit','rate','famges','buerge','wohnzeit','verm','alter','weitkred','wohn','bishkred','beruf','pers','telef','gastarb']
SG_CAT=['laufkont','moral','verw','sparkont','famges','buerge','weitkred','wohn','pers','telef','gastarb']
def load(name):
    receipts=json.loads((HERE/'source_download_receipts.json').read_text())
    r=next(r for r in receipts if r['dataset']==name)
    if name=='LendingClub2017':
        path=HERE/'private/source_raw/LC_Loan_sample_2017.csv';assert sha(path)==r['sha256']
        df=pd.read_csv(path,low_memory=False);keep=df.loan_status.isin(['Fully Paid','Charged Off'])&(pd.to_numeric(df.loan_amnt,errors='coerce')>0);df=df[keep].reset_index(drop=True)
        assert set(pd.to_datetime(df.issue_d,format='%b-%Y',errors='raise').dt.year)=={2017},'source cohort mismatch'
        x=df[LC_COLS].copy()
        for c in ['int_rate','revol_util']:x[c]=pd.to_numeric(x[c].astype(str).str.rstrip('%'),errors='coerce')
        y=(df.loan_status=='Charged Off').to_numpy(np.uint8);exposure=df.loan_amnt.to_numpy(float)
        known={c:df[c].to_numpy() for c in ['id','member_id'] if c in df}
    elif name=='SouthGermanCredit':
        path=HERE/'private/source_raw/SouthGermanCredit.zip';assert sha(path)==r['sha256']
        z=zipfile.ZipFile(path);asc=next(n for n in z.namelist() if n.endswith('SouthGermanCredit.asc'))
        # Coding verified from original author's Report-2019-004 Appendix A: levels(credit_risk)=bad,good in ascending order.
        df=pd.read_csv(io.BytesIO(z.read(asc)),sep=r'\s+');assert len(df)==1000 and set(df.kredit.unique())=={0,1}
        x=df[SG_COLS].copy()
        for c in SG_CAT:x[c]=x[c].astype(str)
        y=(1-df.kredit).to_numpy(np.uint8);exposure=df.hoehe.to_numpy(float);known=None
    else:raise ValueError(name)
    assert np.isfinite(exposure).all() and np.all(exposure>0)
    return x,y,exposure,groups(x,known),r

def groups(x,known=None):
    n=len(x);parent=np.arange(n)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    # Hash collisions are checked against exact covariate tuple equality.
    seen={}
    for i,t in enumerate(x.fillna('<MISSING>').astype(str).itertuples(index=False,name=None)):
        if t in seen:union(i,seen[t])
        else:seen[t]=i
    if known is not None:
        for values in (known.values() if isinstance(known,dict) else [known]):
            seen={}
            for i,k in enumerate(values):
                if pd.isna(k) or str(k).strip().lower() in {'','nan','none','<na>'}:continue
                if k in seen:union(i,seen[k])
                else:seen[k]=i
    return np.array([find(i) for i in range(n)])

def split(x,group,seed,name):
    # Pre-outcome amendment v2: T and development are fixed across replay seeds.
    rng=np.random.default_rng(seed);g=np.random.default_rng(51000).permutation(np.unique(group));n=len(g);dev_g=g[:int(.2*n)];test_g=g[int(.2*n):int(.4*n)];replay_g=g[int(.4*n):]
    dev=np.flatnonzero(np.isin(group,dev_g));test=np.flatnonzero(np.isin(group,test_g));stream=np.flatnonzero(np.isin(group,replay_g));stream=stream[rng.permutation(len(stream))]
    initial_pool=stream[:len(stream)//4]
    if name=='LendingClub2017':score=.04*pd.to_numeric(x.dti,errors='coerce').fillna(20).to_numpy()-.01*pd.to_numeric(x.fico_range_low,errors='coerce').fillna(680).to_numpy()
    else:score=pd.to_numeric(x.laufzeit).to_numpy()+.001*pd.to_numeric(x.hoehe).to_numpy()
    initial=initial_pool[np.lexsort((initial_pool,score[initial_pool]))[:len(initial_pool)//2]]
    batches=np.array_split(stream[len(initial_pool):],10)
    for a,b in [(dev,test),(dev,stream),(test,stream)]:assert not np.intersect1d(group[a],group[b]).size
    return initial,dev,test,batches

def preprocess(x,initial):
    num=[c for c in x if pd.api.types.is_numeric_dtype(x[c])];cat=[c for c in x if c not in num]
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),num),('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    pre.fit(x.iloc[initial]);z=pre.transform(x);return np.c_[np.ones(len(z)),z],pre
