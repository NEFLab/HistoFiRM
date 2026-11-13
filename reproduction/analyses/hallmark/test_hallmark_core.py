import numpy as np,pandas as pd,json,tempfile
from pathlib import Path
from sklearn.linear_model import Ridge
from hallmark_core import *
def run_tests():
    rng=np.random.default_rng(71);checks=[]
    def check(name,test):
        assert test,name
        checks.append(name)
    X=rng.normal(size=(60,2));Y=rng.normal(size=(60,7));P=rng.normal(size=(60,7));w=rng.integers(0,5,60);idx=np.repeat(np.arange(60),w)
    check('frequency_median_even',frequency_median(np.arange(4.),np.ones(4))==1.5)
    for case in ['finite','missing','constant','near_constant','all_missing']:
        x=X.copy()
        if case=='missing':x[:12,0]=np.nan
        if case=='constant':x[:]=1
        if case=='near_constant':x[:,1]=1+1e-14*x[:,1]
        if case=='all_missing':x[:,0]=np.nan
        ex=x[idx];med=np.array([np.nanmedian(ex[:,j]) if np.isfinite(ex[:,j]).any() else 0 for j in range(2)])
        ex=np.where(np.isfinite(ex),ex,med);mu=ex.mean(0);sd=ex.std(0);keep=sd>1e-12
        xx=(ex[:,keep]-mu[keep])/sd[keep];tt=(np.where(np.isfinite(x),x,med)[:,keep]-mu[keep])/sd[keep]
        expected=np.column_stack([Ridge(alpha=10).fit(xx,Y[idx,j]).predict(tt) for j in range(7)]) if keep.any() else np.tile(Y[idx].mean(0),(60,1))
        check('multioutput_expanded_equivalence_'+case,np.max(np.abs(ridge_predict(x,Y,w,x)-expected))<1e-10)
    check('weighted_corr_expanded',np.allclose(corr_cols(Y,P,w),[np.corrcoef(Y[idx,j],P[idx,j])[0,1] for j in range(7)]))
    groups=[np.arange(0,20),np.arange(20,40),np.arange(40,60)]
    ww=counts_for(groups,60,rng_for('p2','TOY'));check('stratum_size',all(ww[g].sum()==len(g) for g in groups))
    check('stable_cancer_stream',np.array_equal(counts_for(groups,60,rng_for('p2','TOY')),ww))
    check('independent_cohort_stream',not np.array_equal(ww,counts_for(groups,60,rng_for('raw','TOY'))))
    check('metric_valid_count',interval(np.array([np.nan]*4),4)[2:]==(0,'valid_resamples_below_threshold'))
    check('minimum_support',np.isnan(ridge_predict(X,Y,np.r_[np.ones(7),np.zeros(53)],X)).all())
    for name,fn in [('invalid_fold',lambda:nuisance_splits(np.repeat(99,60),np.repeat(0,60))),('invalid_r',lambda:fisher_z(1.1)),('duplicate_reference',lambda:unique_keys(pd.DataFrame({'cancer':['a','a'],'target_id':['x','x']}),['cancer','target_id']))]:
        try:fn()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    outer=np.zeros(60,int);inner=np.repeat(range(3),20);splits=nuisance_splits(outer,inner)
    a=p2_all(Y,P,X,splits,ww)
    expected=np.column_stack([p2_all(Y[:,j:j+1],P[:,j:j+1],X,splits,ww)[0] for j in range(7)]).T
    check('all_targets_same_draw_equivalence',np.allclose(a,expected,atol=1e-11))
    check('paired_delta',np.allclose(a[:,3],a[:,2]-a[:,0]))
    try:p2_all(Y,P,X,[(np.array([100]),np.array([0]))],ww)
    except IndexError:checks.append('structural_exception_propagates')
    else:raise AssertionError('swallowed_error')
    with tempfile.TemporaryDirectory() as tmp:
        ref=pd.DataFrame({'cancer':['a'],'target_id':['x'],'r':[.1],'n':[30]});new=ref.copy();new['r']=.2
        path=Path(tmp)/'audit.csv'
        try:verify_reference(new,ref,['r'],path)
        except RuntimeError:check('failure_audit_persisted',path.exists())
        else:raise AssertionError('reproduction_gate')
    return {'status':'PASS','count':len(checks),'checks':checks}
if __name__=='__main__':print(json.dumps(run_tests(),indent=2))
