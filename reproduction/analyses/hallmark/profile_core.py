"""Patient-bootstrap summaries of fixed programme-predictability profiles."""
import numpy as np
from scipy.stats import rankdata

def rank_correlation(x,y):
    x,y=np.broadcast_arrays(x,y)
    valid=np.isfinite(x).all(axis=-1)&np.isfinite(y).all(axis=-1)
    a=rankdata(x,axis=-1,method='average');b=rankdata(y,axis=-1,method='average')
    a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
    result=np.full(den.shape,np.nan,dtype=float)
    np.divide((a*b).sum(axis=-1),den,out=result,where=valid&(den>0))
    return result

def centred_fisher(r):
    good=np.isfinite(r).all(axis=(-2,-1))&(np.abs(r)<1).all(axis=(-2,-1))
    # Endpoint values invalidate the relevant summary; never clip r.
    z=np.full(r.shape,np.nan,dtype=float)
    np.arctanh(r,out=z,where=np.abs(r)<1)
    z=z-z.mean(axis=-1,keepdims=True)-z.mean(axis=-2,keepdims=True)+z.mean(axis=(-2,-1),keepdims=True)
    return np.where(good[...,None,None],z,np.nan)

def summary_schema(models,names):
    rows=[]
    for co in ['TCGA','CPTAC']:
        for model in models[1:]:rows.append({'summary_id':f'median::{co}::{model}','family':'within_cancer_median','cohort':co,'model_id':model,'model':names[model],'profile':'Raw','n_cancers':16 if co=='TCGA' else 8,'n_programmes':50})
    for co in ['TCGA','CPTAC']:
        for model in models[1:]:rows.append({'summary_id':f'centered::{co}::{model}','family':'centered_cross_model','cohort':co,'model_id':model,'model':names[model],'profile':'Centered','n_cancers':8,'n_programmes':50})
    for model in models:
        for kind in ['Raw','Centered']:rows.append({'summary_id':f'external::{kind}::{model}','family':'cross_cohort','cohort':'TCGA-CPTAC','model_id':model,'model':names[model],'profile':kind,'n_cancers':8,'n_programmes':50})
    return rows

def summaries(raw,cancers,common):
    # raw: dict of cohort -> [replicate, cancer, model, programme].
    results=[];shared={};centred={}
    for co in ['TCGA','CPTAC']:
        arr=raw[co]
        rho=rank_correlation(arr[:,:,0:1,:],arr[:,:,1:,:])
        med=np.median(rho,axis=1) # any invalid cancer invalidates its fixed-set median
        results.extend([med[:,j] for j in range(med.shape[1])])
        index=[cancers[co].index(c) for c in common]
        shared[co]=np.transpose(arr[:,index,:,:],(0,2,1,3))
        centred[co]=centred_fisher(shared[co])
    for co in ['TCGA','CPTAC']:
        arr=centred[co].reshape(len(centred[co]),7,400)
        rho=rank_correlation(arr[:,0:1,:],arr[:,1:,:])
        results.extend([rho[:,j] for j in range(6)])
    for m in range(7):
        for kind in [shared,centred]:
            results.append(rank_correlation(kind['TCGA'][:,m].reshape(-1,400),kind['CPTAC'][:,m].reshape(-1,400)))
    return np.stack(results,axis=1)

def resampling_weights(strata,reps,rng):
    strata=np.asarray(strata);w=np.zeros((reps,len(strata)),dtype=np.float64)
    for k in np.unique(strata):
        ix=np.flatnonzero(strata==k)
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=reps)
        assert np.all(w[:,ix].sum(axis=1)==len(ix))
    assert np.all(w.sum(axis=1)==len(strata))
    return w

def patient_pearson(y,p,w):
    # y [n,t], p [n,m,t], w [b,n]; affine preprocessing preserves Pearson.
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.atleast_2d(w)
    yc=y-y.mean(axis=0);pc=p-p.mean(axis=0)
    ys=yc.std(axis=0);ps=pc.std(axis=0)
    valid=(ys[None,:]>0)&(ps>0)
    yn=np.divide(yc,ys,out=np.zeros_like(yc),where=ys>0)
    pn=np.divide(pc,ps,out=np.zeros_like(pc),where=ps>0)
    n=w.sum(axis=1)[:,None,None]
    sy=(w@yn)[:,None,:];sy2=(w@(yn*yn))[:,None,:]
    sp=(w@pn.reshape(len(y),-1)).reshape(len(w),p.shape[1],p.shape[2])
    sp2=(w@(pn*pn).reshape(len(y),-1)).reshape(sp.shape)
    syp=(w@(yn[:,None,:]*pn).reshape(len(y),-1)).reshape(sp.shape)
    vy=sy2-sy*sy/n;vp=sp2-sp*sp/n;cov=syp-sy*sp/n
    denom=np.sqrt(np.maximum(vy,0)*np.maximum(vp,0))
    result=np.full_like(cov,np.nan)
    np.divide(cov,denom,out=result,where=(vy>0)&(vp>0)&valid[None,:,:])
    return result
