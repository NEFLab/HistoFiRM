"""Fixed-predictor Hallmark patient bootstrap; finite target matrices required."""
import hashlib,json
import numpy as np
EPS=1e-12
def rng_for(branch,cancer,seed=20261109):
    h=hashlib.sha256(f'{seed}|{branch}|{cancer}'.encode()).digest()
    return np.random.default_rng(np.frombuffer(h[:16],dtype='<u4'))
def counts_for(groups,n,rng):
    w=np.zeros(n,dtype=np.int64)
    for ix in groups:
        w+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=n)
    return w
def corr_cols(a,b,w):
    a=np.asarray(a,float);b=np.asarray(b,float);w=np.asarray(w,float)
    if a.shape!=b.shape or a.ndim!=2 or len(w)!=len(a):raise ValueError('Correlation axis mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite target matrix')
    sw=w.sum()
    if sw<3:return np.full(a.shape[1],np.nan)
    da=a-np.einsum('n,nj->j',w,a)/sw;db=b-np.einsum('n,nj->j',w,b)/sw
    va=np.einsum('n,nj,nj->j',w,da,da);vb=np.einsum('n,nj,nj->j',w,db,db)
    good=(va/sw>=EPS**2)&(vb/sw>=EPS**2)
    out=np.full(a.shape[1],np.nan)
    out[good]=np.einsum('n,nj,nj->j',w,da,db)[good]/np.sqrt(va[good]*vb[good])
    if np.any(np.abs(out[np.isfinite(out)])>1+1e-12):raise ArithmeticError('Correlation outside [-1,1]')
    return np.clip(out,-1,1)
def frequency_median(x,w):
    ok=np.isfinite(x)&(w>0);x=x[ok];w=w[ok].astype(np.int64)
    if not len(x):return 0.
    order=np.argsort(x);x=x[order];cw=np.cumsum(w[order]);n=int(cw[-1])
    return float((x[np.searchsorted(cw,(n-1)//2+1)]+x[np.searchsorted(cw,n//2+1)])/2)
def preprocess(X,w,Xtest):
    X=np.asarray(X,float);Xtest=np.asarray(Xtest,float);w=np.asarray(w,float)
    if np.isinf(X).any() or np.isinf(Xtest).any():raise ValueError('Infinite covariate')
    if np.any(w<0) or not np.all(w==np.floor(w)) or w.sum()==0:raise ValueError('Invalid frequency weights')
    med=np.array([frequency_median(X[:,j],w) for j in range(X.shape[1])])
    a=np.where(np.isfinite(X),X,med);t=np.where(np.isfinite(Xtest),Xtest,med)
    mu=np.einsum('n,nj->j',w,a)/w.sum();sd=np.sqrt(np.einsum('n,nj,nj->j',w,a-mu,a-mu)/w.sum());keep=sd>EPS
    return (a[:,keep]-mu[keep])/sd[keep],(t[:,keep]-mu[keep])/sd[keep]
def ridge_predict(X,y,w,Xtest,alpha=10.):
    xt,xv=preprocess(X,w,Xtest)
    if w.sum()<max(8,xt.shape[1]+2):return np.full((len(Xtest),y.shape[1]),np.nan)
    ym=np.einsum('n,nj->j',w,y)/w.sum()
    if xt.shape[1]==0:return np.broadcast_to(ym,(len(Xtest),len(ym))).copy()
    xm=np.einsum('n,nj->j',w,xt)/w.sum();xc=xt-xm
    gram=np.einsum('n,ni,nj->ij',w,xc,xc)+alpha*np.eye(xt.shape[1])
    rhs=np.einsum('n,ni,nj->ij',w,xc,y-ym)
    return (xv-xm)@np.linalg.solve(gram,rhs)+ym
def nuisance_splits(outer,inner):
    if not np.isin(outer,range(5)).all() or not np.isin(inner,range(3)).all():raise ValueError('Illegal nuisance fold')
    result=[]
    for of in np.unique(outer):
        oi=np.flatnonzero(outer==of)
        if len(oi)<15 or set(inner[oi])!={0,1,2}:raise ValueError('Invalid eligible stratum')
        for f in range(3):result.append((oi[inner[oi]!=f],oi[inner[oi]==f]))
    return result
def p2_all(Y,P,X,splits,w):
    # One joint patient-frequency vector feeds all 50 targets and both responses.
    both=np.concatenate([Y,P],axis=1);fits=np.full_like(both,np.nan,dtype=float)
    for tr,te in splits:
        fits[te]=ridge_predict(X[tr],both[tr],w[tr],X[te])
    q=Y.shape[1];out=np.full((q,4),np.nan)
    if np.isfinite(fits).all():
        out[:,0]=corr_cols(Y,P,w);out[:,1]=corr_cols(Y,fits[:,:q],w)
        out[:,2]=corr_cols(Y-fits[:,:q],P-fits[:,q:],w)
        out[:,3]=out[:,2]-out[:,0]
        return out
    # Every target has finite inputs by contract; a failed nuisance stratum is statistical NA.
    for j in range(q):
        good=np.isfinite(fits[:,j])&np.isfinite(fits[:,q+j])
        if np.sum(w[good])<3:continue
        y=Y[good,j:j+1];p=P[good,j:j+1];ww=w[good]
        out[j,0]=corr_cols(y,p,ww)[0]
        out[j,1]=corr_cols(y,fits[good,j:j+1],ww)[0]
        out[j,2]=corr_cols(y-fits[good,j:j+1],p-fits[good,q+j:q+j+1],ww)[0]
    out[:,3]=out[:,2]-out[:,0]
    return out
def interval(v,minimum=1900):
    good=np.isfinite(v);n=int(good.sum())
    if n<minimum:return np.nan,np.nan,n,'valid_resamples_below_threshold'
    lo,hi=np.quantile(np.asarray(v)[good],[.025,.975])
    return float(lo),float(hi),n,''
def fisher_z(r):
    r=np.asarray(r,float)
    if np.any(np.abs(r[np.isfinite(r)])>1+1e-12):raise ValueError('Invalid Pearson r')
    return np.arctanh(np.clip(r,-1+1e-12,1-1e-12))
def unique_keys(frame,keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():raise ValueError('Invalid or duplicate reference keys')
def verify_reference(new,ref,metrics,path,tol=1e-8):
    keys=['cancer','target_id'];unique_keys(new,keys);unique_keys(ref,keys)
    m=new.merge(ref[keys+metrics+['n']],on=keys,how='outer',suffixes=('_new','_ref'),indicator=True,validate='one_to_one')
    m['pass']=(m['_merge']=='both')&(m.n_new==m.n_ref)
    for col in metrics:
        a=m[col+'_new'];b=m[col+'_ref'];m[col+'_abs_difference']=(a-b).abs()
        m['pass']&=((a-b).abs()<=tol)|(a.isna()&b.isna())
    m.to_csv(path,index=False)
    if not m['pass'].all():raise RuntimeError(f'Reference reproduction failed: {path}')
    return {'rows':len(m),'maximum_absolute_difference':float(m[[x+'_abs_difference' for x in metrics]].max().max()),'passed':True}
