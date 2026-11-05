"""Fixed-cancer/fixed-target evaluation, not flattened Pearson or independent-target tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .contracts import require


def cell_metrics(y,p,min_n=10):
    y=np.asarray(y,float);p=np.asarray(p,float);n_total=len(y)
    require(len(p)==len(y),'Truth/prediction length mismatch')
    require(np.isfinite(p).all(),'Nonfinite prediction is model failure, not row exclusion')
    good=np.isfinite(y);y=y[good];p=p[good];n=len(y)
    out={'n_total':n_total,'n_valid':n,'truth_eligible':False,'pearson':np.nan,'spearman':np.nan,
         'mae':np.nan,'rmse':np.nan,'nrmse':np.nan,'ccc':np.nan,'r2':np.nan,'reason':''}
    if n==0:out['reason']='NO_TRUTH';return out
    dy=y-y.mean();dp=p-p.mean();vy=np.mean(dy*dy);vp=np.mean(dp*dp);cov=np.mean(dy*dp)
    mse=np.mean((p-y)**2);out.update(mae=float(np.mean(abs(p-y))),rmse=float(np.sqrt(mse)),
        bias=float(p.mean()-y.mean()),truth_mean=float(y.mean()),pred_mean=float(p.mean()),
        truth_sd=float(y.std(ddof=1)) if n>1 else np.nan,pred_sd=float(p.std(ddof=1)) if n>1 else np.nan)
    if vy>0:
        out['r2']=1.-mse/vy
        if n>1:out['nrmse']=np.sqrt(mse)/y.std(ddof=1)
    denom=vy+vp+(p.mean()-y.mean())**2
    if denom>0:out['ccc']=2*cov/denom
    if n<min_n:out['reason']='N_VALID_BELOW_MINIMUM';return out
    if vy==0:out['reason']='CONSTANT_TRUTH';return out
    out['truth_eligible']=True
    if vp==0:out['reason']='CONSTANT_PREDICTION';return out
    out['pearson']=float(np.clip(cov/np.sqrt(vy*vp),-1,1));out['spearman']=float(spearmanr(y,p).statistic)
    return out


def table(y,p,cancer,targets,min_n=10):
    rows=[];cancer=np.asarray(cancer)
    require(y.shape==p.shape==(len(cancer),len(targets)),'Evaluation array shape mismatch')
    for c in sorted(set(cancer)):
        use=cancer==c
        for j,t in enumerate(targets):rows.append(dict(cancer=c,target_id=t,**cell_metrics(y[use,j],p[use,j],min_n)))
    return pd.DataFrame(rows)


def summarize(cells):
    """Undefined model correlations cannot disappear to improve primary macro score."""
    groups=[]
    metrics=['pearson','spearman','mae','rmse','nrmse','ccc','r2']
    for c,g in cells.groupby('cancer',sort=True):
        valid=g[g.truth_eligible.astype(bool)];require(len(valid)>0,'Cancer has no truth-eligible targets')
        complete=np.isfinite(valid.pearson).all()
        r={'cancer':c,'n_patients':int(g.n_total.max()),'n_targets':len(g),'truth_eligible_targets':len(valid),
           'undefined_prediction_correlations':int((~np.isfinite(valid.pearson)).sum()),
           'primary_status':'COMPLETE' if complete else 'INCOMPLETE_CONSTANT_PREDICTION'}
        for k in metrics:
            r[k]=float(valid[k].mean()) if (k not in ['pearson','spearman'] or complete) else np.nan
        r['pearson_estimable_only_descriptive']=float(valid.pearson.mean())
        r['pearson_undefined_as_zero_sensitivity']=float(valid.pearson.fillna(0).mean())
        groups.append(r)
    df=pd.DataFrame(groups);overall={k:float(df[k].mean()) for k in metrics}
    if df.primary_status.ne('COMPLETE').any():overall['pearson']=np.nan;overall['spearman']=np.nan
    overall['pearson_undefined_as_zero_sensitivity']=float(df.pearson_undefined_as_zero_sensitivity.mean())
    return {'overall':overall,'per_cancer':groups,'estimand':'RAW_R_MEAN_TARGET_THEN_FIXED_CANCER',
            'target_support':'TRUTH_ONLY_MIN_N_VARIANCE; no model-specific silent drop'}


def weighted_corr_matrix(y,p,w,min_n=10):
    """w: bootstrap row counts. Stable, fixed origin centering before weighted moments."""
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float)
    good=np.isfinite(y);n=w@good.astype(float);safe_n=np.maximum(n,1.)
    center=np.nanmean(y,axis=0);yp=np.where(good,y-center,0.)
    pp=np.where(good,p-p.mean(axis=0),0.)
    sy=w@yp;sp=w@pp
    yy=w@(yp*yp)-sy*sy/safe_n;vv=w@(pp*pp)-sp*sp/safe_n
    xy=w@(yp*pp)-sy*sp/safe_n
    den=np.sqrt(np.maximum(yy,0)*np.maximum(vv,0))
    out=np.divide(xy,den,out=np.full_like(xy,np.nan),where=den>0)
    out[(n<min_n)|(yy<=0)|(vv<=0)]=np.nan
    return np.clip(out,-1,1)


def paired_bootstrap(y,p1,p0,cancers,B=2000,seed=20260912,min_n=10,batch=32):
    """Fixed cancers/targets; patients sampled together within cancer. No p-value claim."""
    require(B>=20,'B too small');y=np.asarray(y,float);p1=np.asarray(p1,float);p0=np.asarray(p0,float);c=np.asarray(cancers)
    require(y.shape==p1.shape==p0.shape,'Paired arrays mismatch')
    require(np.isfinite(p1).all() and np.isfinite(p0).all(),'Prediction failure, not silently filterable')
    rng=np.random.default_rng(seed);draws=np.zeros(B);valid_draw=np.ones(B,dtype=bool);point=[];nc=len(set(c))
    by_cancer={}
    for cancer in sorted(set(c)):
        yc=y[c==cancer];a=p1[c==cancer];b=p0[c==cancer];n=len(yc)
        support=(np.isfinite(yc).sum(0)>=min_n)&(np.nanstd(yc,axis=0)>0)
        require(support.any(),'No supported targets in cancer')
        yc=yc[:,support];a=a[:,support];b=b[:,support]
        da=weighted_corr_matrix(yc,a,np.ones((1,n)),min_n)[0]
        db=weighted_corr_matrix(yc,b,np.ones((1,n)),min_n)[0]
        require(np.isfinite(da).all() and np.isfinite(db).all(),'Undefined original prediction correlation: CI not estimable')
        dpoint=float(np.mean(da-db));point.append(dpoint);cdraw=np.zeros(B);cvalid=np.ones(B,dtype=bool)
        for s in range(0,B,batch):
            k=min(batch,B-s);w=rng.multinomial(n,np.full(n,1./n),size=k).astype(float)
            ra=weighted_corr_matrix(yc,a,w,min_n);rb=weighted_corr_matrix(yc,b,w,min_n)
            good=np.isfinite(ra).all(1)&np.isfinite(rb).all(1)
            cvalid[s:s+k]=good;cdraw[s:s+k]=np.mean(ra-rb,axis=1)
        draws+=cdraw/nc;valid_draw&=cvalid
        by_cancer[cancer]={'delta':dpoint,'B_valid':int(cvalid.sum()),'ci95':np.quantile(cdraw[cvalid],[.025,.975]).tolist() if cvalid.any() else None}
    valid=int(valid_draw.sum());bad=B-valid
    result={'delta':float(np.mean(point)),'B_requested':B,'B_valid':valid,'B_failed':bad,'seed':seed,
            'uncertainty':'CONDITIONAL_ON_FROZEN_PREDICTIONS','resampling':'PATIENT_WITHIN_FIXED_CANCER_FIXED_TARGETS',
            'new_p_q_FWER':False,'per_cancer_descriptive':by_cancer}
    if bad/B>.01:result.update(status='CI_NOT_ESTIMABLE_TOO_MANY_INVALID_DRAWS',ci95=None)
    else:result.update(status='DIAGNOSTIC_CI_COMPLETE',ci95=np.quantile(draws[valid_draw],[.025,.975]).tolist())
    return result
