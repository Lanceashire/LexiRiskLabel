"""Explicit future-environment controls for later GATE diagnostics (not E4 runs)."""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class FutureProtocol:
    future_batch_size:int=160
    capacity:int=80
    risk_cap:float=20.
    threshold:float=.5
    planning_horizon:int=2
    actual_future_release:bool=True
    future_approval:str='dynamic'
    selection_worlds:int=16
    verification_worlds:int=8

    def __post_init__(self):
        if self.planning_horizon not in [1,2]:raise ValueError('planning horizon')
        if self.future_approval not in ['fixed','dynamic']:raise ValueError('future approval')
        if self.future_batch_size<0 or self.capacity<0 or self.capacity>self.future_batch_size or self.risk_cap<0:raise ValueError('future budget/capacity')

    def planner_parameters(self):
        return {k:v for k,v in asdict(self).items() if k not in ['actual_future_release']}

    def environment_parameters(self):
        return {k:v for k,v in asdict(self).items() if k not in ['planning_horizon','selection_worlds','verification_worlds']}

    def release(self,approved_ids,labels):
        ids=np.asarray(approved_ids,int)
        return (ids,np.asarray(labels)[ids]) if self.actual_future_release else (np.empty(0,int),np.empty(0,int))

    def approvals(self,probability,prices,fixed_ids=None):
        if len(probability)!=self.future_batch_size:raise ValueError('planner/environment future size mismatch')
        candidates=np.asarray(fixed_ids,int) if self.future_approval=='fixed' else np.flatnonzero(np.asarray(probability)<self.threshold)
        ids=[];spent=0.
        for i in candidates[np.argsort(np.asarray(probability)[candidates],kind='stable')]:
            if len(ids)>=self.capacity:break
            if spent+prices[i]<=self.risk_cap+1e-10:ids.append(int(i));spent+=prices[i]
        return np.asarray(ids,int)
