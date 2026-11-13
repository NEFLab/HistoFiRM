import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import sys,json,hashlib,time,platform
import numpy as np,pandas as pd,scipy
from profile_core import resampling_weights,patient_pearson,summaries
from test_profile_core import test
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def main(root):
    started=time.time();cfg=json.loads((root/'config.json').read_text());out=root/'outputs'
    gate=json.loads((root/'POINT_GATE.json').read_text());assert gate['status']=='PASS'
    assert sha(root/'INPUT_MANIFEST.json')==gate['manifest_sha256']
    for item in json.loads((root/'INPUT_MANIFEST.json').read_text()):assert sha(item['path'])==item['sha256']
    assert not (out/'B1_INTERVALS_38.csv').exists()
    tests=test();dump(out/'CORE_TESTS.json',tests)
    axes=json.loads((root/'AXES.json').read_text());reps=cfg['replicates'];models=cfg['models']
    schema=pd.read_csv(out/'SUMMARY_POINT_REPRODUCTION.csv')
    total=sum(len(v) for v in axes['cancers'].values());streams=np.random.SeedSequence(cfg['bootstrap_seed']).spawn(total)
    stream_index=0;raw={};audit=[]
    dump(root/'RUN_STATUS.json',{'status':'RUNNING','replicates':reps,'seed':cfg['bootstrap_seed']})
    for co in ['TCGA','CPTAC']:
        path=root/'private_inputs'/f'{co}.npz'
        manifest=json.loads((root/'INPUT_MANIFEST.json').read_text());pin=next(x for x in manifest if x['path']==str(path));assert sha(path)==pin['sha256']
        with np.load(path,allow_pickle=False) as z:y=z['Y'];p=z['P'];cs=z['cancer'];fold=z['outer']
        raw[co]=np.empty((reps,len(axes['cancers'][co]),7,50),float)
        for ci,c in enumerate(axes['cancers'][co]):
            ix=np.flatnonzero(cs==c);rng=np.random.default_rng(streams[stream_index]);stream_index+=1
            w=resampling_weights(fold[ix],reps,rng);r=patient_pearson(y[ix],p[ix],w)
            raw[co][:,ci]=r
            rec={'cohort':co,'cancer':c,'n':len(ix),'replicates':reps,'rng_spawn_key':list(streams[stream_index-1].spawn_key),'weight_sha256':hashlib.sha256(w.astype('<i4').tobytes()).hexdigest(),'nonfinite_correlations':int((~np.isfinite(r)).sum()),'endpoint_correlations':int((np.abs(r)>=1).sum()),'shared_weights_all_models_and_programmes':True,'strata_n_exact':True}
            audit.append(rec);print(json.dumps({'completed_cancer':rec,'seconds':round(time.time()-started,1)}),flush=True)
            dump(root/'RUN_STATUS.json',{'status':'RUNNING','completed_blocks':len(audit),'total_blocks':total,'elapsed_seconds':time.time()-started})
        np.savez_compressed(root/'private_inputs'/f'{co}_BOOTSTRAP_PEARSON.npz',pearson=raw[co],cancers=np.array(axes['cancers'][co]),models=np.array(models),target_ids=np.array(axes['targets']))
    values=summaries(raw,axes['cancers'],axes['shared_cancers']);assert values.shape==(reps,38)
    rows=[]
    for j,row in schema.iterrows():
        v=values[:,j];valid=np.isfinite(v);n=int(valid.sum());keep=n>=cfg['minimum_valid']
        lo,hi=np.quantile(v[valid],[.025,.975],method='linear') if keep else [np.nan,np.nan]
        rows.append({**row.to_dict(),'lower_95':float(lo),'upper_95':float(hi),'valid_replicates':n,'total_replicates':reps,'bootstrap_mean':float(v[valid].mean()) if n else np.nan,'bootstrap_median':float(np.median(v[valid])) if n else np.nan,'interval_status':'PASS' if keep else 'INSUFFICIENT_VALID','interval':'pointwise percentile 95%','point_outside_percentile_interval':bool(row['estimate']<lo or row['estimate']>hi) if keep else None})
    table=pd.DataFrame(rows);table.to_csv(out/'B1_INTERVALS_38.csv',index=False,float_format='%.17g')
    rep=pd.DataFrame(values,columns=schema.summary_id);rep.insert(0,'replicate',np.arange(1,reps+1));rep.to_csv(out/'B1_SUMMARY_REPLICATES_2000.csv',index=False,float_format='%.17g')
    dump(out/'RESAMPLING_AUDIT.json',audit)
    dump(out/'RUNTIME.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'elapsed_seconds':time.time()-started,'threads_per_blas':2})
    allfiles=list((root/'private_inputs').glob('*BOOTSTRAP*'))+list(out.iterdir())+[root/'config.json',root/'POINT_GATE.json',root/'INPUT_MANIFEST.json',root/'AXES.json']+list(root.glob('*.py'))
    dump(out/'RESULT_MANIFEST.json',[{'path':str(q),'bytes':q.stat().st_size,'sha256':sha(q)} for q in allfiles])
    status={'status':'COMPLETE' if (table.interval_status=='PASS').all() else 'PARTIAL','intervals_pass':int((table.interval_status=='PASS').sum()),'intervals_total':38,'valid_replicates_min':int(table.valid_replicates.min()),'point_outside_percentile_interval':int(table.point_outside_percentile_interval.sum()),'elapsed_seconds':time.time()-started,'no_retraining':True,'no_new_p_values':True}
    dump(root/'RUN_STATUS.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='S37 fixed-predictor profile bootstrap')
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    try:main(root)
    except Exception as e:
        dump(root/'RUN_STATUS.json',{'status':'FAILED','error_type':type(e).__name__,'error':str(e)});raise
