"""Role-separated, identity-checked diagnostic runtime; frozen v2 code is imported.

Acquisition receives D labels and O/C/Q features only. The evaluator is the sole
recipient of outer T features/labels. Individual records remain in private/.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,time,sys,subprocess,platform
import numpy as np
import pandas as pd
from scipy.special import expit
from xgboost import XGBClassifier
import scipy,sklearn,xgboost

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/research_v2'))
import core as frozen
import run_real as legacy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    if isinstance(value,np.ndarray):return frozen.digest(value)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=json_default).encode()).hexdigest()


def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=json_default)+'\n');temporary.replace(path)


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(row,ensure_ascii=False,default=json_default)+'\n' for row in rows));tmp.replace(path)


def run_identity(config):
    paths=[HERE/x for x in ['runtime.py','run_e1.py','audit.py','gate_protocol.py','protocol.json']]
    paths+=[ROOT/'experiments/research_v2'/x for x in ['core.py','run_real.py','config.json','run_mechanisms.py']]
    code={str(p.relative_to(ROOT)):sha(p) for p in paths}
    data={name:sha(ROOT/'data/research_v2'/f'{name}.csv') for name in config['datasets']}
    anchors={}
    for dataset in config['datasets']:
        for seed in config['seeds']:
            base=ROOT/'results/research_v2/final_real'/f'{dataset}_{seed}_fixed_MinRisk'
            for r in [1]+config['anchor_rounds']:
                p=base/f'round_{r:02d}_selection.json';anchors[str(p.relative_to(ROOT))]=sha(p)
            for name in ['identity.json','test_predictions.npz']:
                p=base/name;anchors[str(p.relative_to(ROOT))]=sha(p)
    result={'base_commit':config['base_commit'],'configuration':config,'config_hash':digest(config),'code':code,'datasets':data,'anchors':anchors,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
            'source_protocol_hash':sha(HERE/'experiment_protocol_zh.md')}
    result['identity_hash']=digest(result);return result


def verify_identity(stored,current):
    if stored!=current:raise RuntimeError('identity mismatch: configuration, code, data or anchor changed')


@dataclass
class FeedbackService:
    """Receives labels; has no path back into selector arguments."""
    d_ids:np.ndarray
    o_ids:np.ndarray
    c_ids:np.ndarray
    d_x:np.ndarray
    o_x:np.ndarray
    c_x:np.ndarray
    _y:np.ndarray
    private_directory:Path
    state_id:str

    def lock(self,selected,method,selector_seed):
        selected=np.asarray(selected,int)
        if len(np.unique(selected))!=len(selected):raise AssertionError('duplicate collection')
        if not set(selected)<=set(range(len(self.c_ids))):raise AssertionError('outside current candidate pool')
        ids=np.sort(self.c_ids[selected]);key=digest({'state_id':self.state_id,'selected_ids':ids})
        path=self.private_directory/'selection_locks'/f'{key}.json'
        record={'state_id':self.state_id,'method':method,'selector_seed':selector_seed,'selected_ids':ids,'selected_hash':digest(ids),'label_release_state':'locked_before_reveal'}
        if not path.exists():write_json(path,record)
        return key

    def reveal(self,selected,key):
        p=self.private_directory/'selection_locks'/f'{key}.json'
        if not p.is_file():raise RuntimeError('labels requested before selection lock')
        selected=np.asarray(selected,int);ids=np.r_[self.d_ids,self.o_ids,self.c_ids[selected]]
        xx=np.vstack([self.d_x,self.o_x,self.c_x[selected]])
        if len(np.unique(ids))!=len(ids):raise AssertionError('already-observed label collected again')
        order=np.argsort(ids,kind='stable');return ids[order],xx[order],self._y[ids[order]]


@dataclass
class EvaluationService:
    _ids:np.ndarray
    _x:np.ndarray
    _y:np.ndarray
    _target:np.ndarray
    private_directory:Path

    def predict(self,model,learner):
        return expit(self._x@model) if learner=='LR' else model.predict_proba(self._x[:,1:].astype(np.float32))[:,1]

    def evaluate(self,prediction):
        e=(prediction-self._y)**2;target=self._target
        return {'brier':float(e.mean()),'target_brier':float(e[target].mean()) if target.any() else None,
                'rest_brier':float(e[~target].mean()) if (~target).any() else None,'target_share':float(target.mean()),'test_n':len(e)}

    def save(self,key,prediction):
        dest=self.private_directory/'predictions'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():np.savez_compressed(dest,ids=self._ids,y=self._y,target=self._target,prediction=prediction,squared_error=(prediction-self._y)**2)


@dataclass
class Context:
    state_id:str
    dataset:str
    seed:int
    round:int
    acquisition:frozen.State
    feedback:FeedbackService
    evaluation:EvaluationService
    manifest:dict
    initial_ids:np.ndarray


def build_context(dataset,seed,round_id,config,private_directory,label_override=None):
    raw,y,exposure,data_hash,columns=legacy.load(dataset)
    if label_override is not None:y=np.asarray(label_override,int)
    directory=ROOT/'results/research_v2/final_real'/f'{dataset}_{seed}_fixed_MinRisk'
    log=json.loads((directory/f'round_{round_id:02d}_selection.json').read_text())
    original=json.loads((directory/'identity.json').read_text())
    if data_hash!=original['dataset_sha256']:raise RuntimeError('anchor dataset mismatch')
    first=json.loads((directory/'round_01_selection.json').read_text());initial=np.array(first['visible_ids'],int)
    did,oid,cid,qid=[np.asarray(log[k],int) for k in ['visible_ids','ordinary_ids','candidate_ids','reference_ids']]
    for a,b in [(did,oid),(did,cid),(oid,cid)]:assert not np.intersect1d(a,b).size
    # Fit and transform acquisition-visible parts only. T is handled separately below.
    pre=legacy.preprocessor(raw);pre.fit(raw.iloc[initial])
    def transform(ids):return np.c_[np.ones(len(ids)),np.asarray(pre.transform(raw.iloc[ids]),float)]
    dx,ox,cx,qx=map(transform,[did,oid,cid,qid])
    theta=frozen.fit_logistic(dx,y[did],config['lr_ridge'])
    a=frozen.State(dx,y[did],ox,cx,qx,np.asarray(log['candidate_costs'],float),float(log['budget']),theta,seed*100+round_id,
                   config['lr_ridge'],original['config']['partners'],original['config']['exchanges'])
    state_id=f'{dataset}_seed{seed}_round{round_id}'
    prediction_record=np.load(directory/'test_predictions.npz');tid=prediction_record['ids'];mask=prediction_record['target']
    assert not np.intersect1d(tid,np.r_[did,oid,cid,qid]).size
    assert not np.intersect1d(qid,np.r_[did,oid,cid]).size
    evaluator=EvaluationService(tid,transform(tid),y[tid],mask,Path(private_directory))
    feedback=FeedbackService(did,oid,cid,dx,ox,cx,y,Path(private_directory),state_id)
    representation=digest({'features':pre.get_feature_names_out().tolist(),'initial_x':digest(transform(initial)),
                           'D':digest(dx),'O':digest(ox),'C':digest(cx),'Q_ref':digest(qx)})
    metadata={'state_id':state_id,'dataset':dataset,'input_hash':data_hash,'protocol':'same_state_development_fork','anchor_policy':'MinRisk','seed':seed,'round':round_id,
              'D_hash':digest(did),'O_hash':digest(oid),'C_hash':digest(cid),'reference_hash':digest(qid),
              'preprocessor_hash':digest({'features':pre.get_feature_names_out().tolist(),'initial_x':digest(transform(initial))}),
              'price_hash':digest({'r0_parameters':original['config']['model'],'D0_hash':digest(initial),'dataset_hash':data_hash}),
              'cost_hash':digest(a.cost),'B':a.budget,'label_release_state':config['label_release'],'representation_hash':representation,
              'ordinary_n':len(oid),'candidate_n':len(cid),'initial_visible_n':len(initial),'current_visible_n':len(did),
              'config_hash':digest(config)}
    return Context(state_id,dataset,seed,round_id,a,feedback,evaluator,metadata,initial)


class FitCache:
    def __init__(self,config,code_hash,private_directory):
        self.config=config;self.code_hash=code_hash;self.private=Path(private_directory);self.entries={};self.fits=0;self.hits=0

    def fit(self,context,selected,learner,lock_key):
        ids,xx,yy=context.feedback.reveal(selected,lock_key);fit_seed=context.seed*1000+context.round
        params={'ridge':self.config['lr_ridge']} if learner=='LR' else self.config['xgb']
        key=digest({'dataset':context.dataset,'train_ids':digest(ids),'x':digest(xx),'y':digest(yy),'learner':learner,
                    'params':params,'fit_seed':fit_seed,'code':self.code_hash,'representation':context.manifest['representation_hash']})
        if key in self.entries:
            self.hits+=1;return {**self.entries[key],'cache_hit':True,'charged_fit_seconds':0.,'fit_key':key}
        started=time.perf_counter()
        if learner=='LR':
            model=frozen.fit_logistic(xx,yy,self.config['lr_ridge']);backend='cpu';updater='newton-cholesky'
        else:
            model=XGBClassifier(**self.config['xgb'],random_state=fit_seed,objective='binary:logistic',eval_metric='logloss')
            model.fit(xx[:,1:].astype(np.float32),yy)
            actual=json.loads(model.get_booster().save_config())['learner'];backend=actual['generic_param']['device'];updater=actual['gradient_booster']['gbtree_train_param']['updater']
            if not backend.startswith('cuda'):raise RuntimeError('actual GPU backend required, observed '+backend)
        fit_seconds=time.perf_counter()-started;prediction=context.evaluation.predict(model,learner)
        result={'prediction':prediction,'metrics':context.evaluation.evaluate(prediction),'actual_backend':backend,'updater':updater,
                'fit_seconds':fit_seconds,'cache_hit':False,'charged_fit_seconds':fit_seconds,'fit_key':key,'train_n':len(ids),'train_hash':digest(ids)}
        self.entries[key]=result;self.fits+=1;context.evaluation.save(key,prediction)
        return result


def bootstrap_ig(state,config):
    rng=np.random.default_rng(state.seed);predictions=[];started=time.perf_counter()
    for member in range(config['committee_members']):
        sample=rng.integers(0,len(state.visible_y),len(state.visible_y));yy=state.visible_y[sample]
        if len(np.unique(yy))<2:predictions.append(np.full(len(state.cost),yy.mean()));continue
        model=XGBClassifier(**config['xgb'],random_state=state.seed+member,objective='binary:logistic',eval_metric='logloss')
        model.fit(state.visible_x[sample,1:].astype(np.float32),yy)
        actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if not actual.startswith('cuda'):raise RuntimeError('IG GPU backend mismatch')
        predictions.append(model.predict_proba(state.candidate_x[:,1:].astype(np.float32))[:,1])
    p=np.asarray(predictions);score=np.maximum(frozen.entropy(p.mean(axis=0))-frozen.entropy(p).mean(axis=0),0)
    return score,time.perf_counter()-started


def select_e1(state,method,config,draw=0,utility=None):
    if method=='FullFeedback':
        ids=np.arange(len(state.cost));return frozen.Selection(ids,float(state.cost.sum()),{'diagnostic_only':True,'selection_seconds':0.})
    if method=='Random-feasible':
        seed=state.seed*1000+811+draw;start=time.perf_counter();ids=frozen.random_fill(state.cost,state.budget,np.random.default_rng(seed))
        return frozen.finish(state,ids,{'selection_seconds':time.perf_counter()-start,'random_draw':draw,'selector_seed':seed})
    return frozen.select(method,state,utility)
