"""Training-only support. Structural missing labels do not cancel an entire cancer.
Unsupported outputs remain NaN; training masks never edit observed truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from common import require

@dataclass
class TargetScale:
    count: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    supported: np.ndarray
    variable: np.ndarray
    @classmethod
    def fit(cls,y,min_count=2):
        y=np.asarray(y,dtype=np.float64);require(y.ndim==2 and not np.isinf(y).any(),'Invalid labels')
        finite=np.isfinite(y);n=finite.sum(0);mean=np.where(finite,y,0.).sum(0)/np.maximum(n,1)
        v=np.where(finite,y-mean,0.);sd=np.sqrt((v*v).sum(0)/np.maximum(n,1))
        supported=n>=min_count;variable=supported&(sd>1e-12)
        return cls(n,mean,np.where(variable,sd,1.),supported,variable)
    def transform(self,y):
        y=np.asarray(y,dtype=np.float64);require(y.shape[-1]==len(self.mean),'Target dimension differs')
        z=(y-self.mean)/self.sd;z[:,~self.variable]=np.nan;return z
    def inverse(self,p):
        p=np.asarray(p,dtype=np.float64);require(p.shape[-1]==len(self.mean),'Prediction width differs')
        q=p*self.sd+self.mean;q[:,self.supported&~self.variable]=self.mean[self.supported&~self.variable]
        q[:,~self.supported]=np.nan;return q
    def as_dict(self):return {k:np.asarray(getattr(self,k)) for k in ('count','mean','sd','supported','variable')}
    @classmethod
    def from_dict(cls,x): return cls(**{k:np.asarray(x[k]) for k in ('count','mean','sd','supported','variable')})

def masked_patient_mse(pred,target):
    import torch
    good=torch.isfinite(target);n=good.sum(-1);ok=n>0
    require(bool(ok.any()),'This batch contains no supported observed labels')
    require(bool(torch.isfinite(pred[good]).all()),'Nonfinite predictions at labelled outputs')
    safe=torch.where(good,target,torch.zeros_like(target))
    sq=torch.where(good,(pred-safe)**2,torch.zeros_like(pred))
    return (sq.sum(-1)[ok]/n[ok]).mean()

def comparison_support(counts_by_fold,min_count=2):
    a=np.asarray(counts_by_fold);require(a.ndim==2 and a.shape[0]==5,'Need five pre-evaluation training supports')
    return np.all(a>=min_count,axis=0)

def partial_ridge_fit_predict(xfit,yfit,xnew,alpha=100.):
    """Pure cancer-specific auxiliary reference. No global/zero label imputation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    xfit=np.asarray(xfit);yfit=np.asarray(yfit,dtype=float);xnew=np.asarray(xnew)
    require(np.isfinite(xfit).all() and np.isfinite(xnew).all(),'Nonfinite X')
    xs=StandardScaler().fit(xfit);a=xs.transform(xfit);b=xs.transform(xnew);s=TargetScale.fit(yfit)
    out=np.full((len(xnew),yfit.shape[1]),np.nan);fits={}
    groups={}
    for j in np.flatnonzero(s.variable):
        m=np.isfinite(yfit[:,j]);groups.setdefault(np.packbits(m).tobytes(),[m,[]])[1].append(j)
    for m,cols in groups.values():
        ys=(yfit[np.ix_(m,cols)]-s.mean[cols])/s.sd[cols]
        model=Ridge(alpha=alpha,fit_intercept=True).fit(a[m],ys)
        out[:,cols]=np.asarray(model.predict(b)).reshape(len(b),len(cols))*s.sd[cols]+s.mean[cols]
        fits[tuple(cols)]=model
    const=s.supported&~s.variable;out[:,const]=s.mean[const]
    return out,{'x_scaler':xs,'target_scale':s,'fits':fits}
