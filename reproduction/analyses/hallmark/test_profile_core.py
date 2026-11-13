from pathlib import Path
import numpy as np,json
from scipy.stats import spearmanr,pearsonr
from profile_core import patient_pearson,resampling_weights,rank_correlation,centred_fisher
def test():
    rng=np.random.default_rng(14);y=rng.normal(size=(37,5));p=rng.normal(size=(37,3,5))
    strata=np.repeat([0,1,2],[11,13,13]);w=resampling_weights(strata,8,rng)
    r=patient_pearson(y,p,w);err=0
    for b in range(8):
        ix=np.repeat(np.arange(37),w[b].astype(int))
        for m in range(3):
            for t in range(5):err=max(err,abs(r[b,m,t]-pearsonr(y[ix,t],p[ix,m,t]).statistic))
    assert err<1e-12
    order=rng.permutation(37);rr=patient_pearson(y[order],p[order],w[:,order]);order_err=float(np.max(abs(r-rr)));assert order_err<1e-12
    a=np.array([1,2,2,4,9]);b=np.array([2,4,4,1,3]);tieerr=float(abs(rank_correlation(a,b)-spearmanr(a,b).statistic));assert tieerr<1e-12
    p[:,1,3]=7;constant=patient_pearson(y,p,w);assert np.isnan(constant[:,1,3]).all()
    z=np.array([[[.2,.3,.4],[.4,.2,.7]],[[.1,1,.3],[.3,.4,.5]]]);ctr=centred_fisher(z)
    assert np.isnan(ctr[1]).all();centererr=float(max(abs(ctr[0].mean(axis=0)).max(),abs(ctr[0].mean(axis=1)).max()));assert centererr<1e-12
    assert np.isnan(rank_correlation(np.ones(5),b))
    return {'status':'PASS','weighted_vs_explicit_patient_repetition_max_error':float(err),'row_permutation_max_error':order_err,'tied_rank_scipy_max_error':tieerr,'centered_margin_max_error':centererr,'constant_input_invalidated':True,'fisher_endpoint_invalidated_without_clipping':True,'stratum_sample_sizes_preserved':True}
if __name__=='__main__':print(json.dumps(test(),indent=2))
