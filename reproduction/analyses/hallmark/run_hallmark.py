from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,time
from hallmark_core import *
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    root=Path.cwd();gate=json.loads((root/'POINT_GATE.json').read_text());assert gate['status']=='PASS'
    assert sha(root/'INPUT_MANIFEST.json')==gate['manifest_sha256']
    for row in json.loads((root/'INPUT_MANIFEST.json').read_text()):
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Input or implementation changed: '+row['path'])
    cfg=json.loads((root/'config.json').read_text());B=cfg['statistics']['bootstrap_replicates'];assert B==2000
    out=root/'outputs';repdir=out/'replicates';repdir.mkdir(exist_ok=True)
    method='pointwise percentile 95%; patient bootstrap conditional on fixed predictions'
    rawrows=[];macrorows=[];p2rows=[];externalrows=[];data={};rawreps={};start=time.time()
    def save_or_compute(name,shape,fun):
        p=repdir/(name+'.npz');receipt=p.with_suffix('.sha256')
        if p.exists():
            assert receipt.exists() and receipt.read_text()==sha(p),'Incomplete checkpoint'
            a=np.load(p)['values'];assert a.shape==shape;return a
        a=np.full(shape,np.nan);fun(a)
        tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,values=a);tmp.replace(p);receipt.write_text(sha(p));return a
    for cohort in ['TCGA','CPTAC']:
        with np.load(root/'private_inputs'/f'{cohort}.npz') as z:d={k:z[k] for k in z.files}
        data[cohort]=d;pts=pd.read_csv(out/f'{cohort}_POINTS.csv').set_index(['cancer','target_id'])
        for cancer in cfg['scope'][cohort.lower()+'_cancers']:
            ix=np.flatnonzero(d['cancer']==cancer);Y=d['Y'][ix];P=d['P'][ix];n=len(ix)
            groups=[np.flatnonzero(d['outer'][ix]==k) for k in range(5)] if cohort=='TCGA' else [np.arange(n)]
            rng=rng_for('raw_'+cohort,cancer)
            def compute(a):
                for b in range(B):a[b]=corr_cols(Y,P,counts_for(groups,n,rng))
            v=save_or_compute('raw_'+cohort+'_'+cancer,(B,50),compute);rawreps[(cohort,cancer)]=v
            for j,tar in enumerate(d['targets']):
                lo,hi,nv,reason=interval(v[:,j]);rawrows.append({'cohort':cohort,'cancer':cancer,'target_id':tar,'n':n,'estimate':pts.loc[(cancer,tar),'r'],'ci_low':lo,'ci_high':hi,'valid_resamples':nv,'attempted_resamples':B,'na_reason':reason,'interval_method':method})
            lo,hi,nv,reason=interval(v.mean(1));macrorows.append({'cohort':cohort,'cancer':cancer,'n':n,'targets':50,'estimate':float(pts.loc[cancer].r.mean()),'ci_low':lo,'ci_high':hi,'valid_resamples':nv,'attempted_resamples':B,'na_reason':reason})
            print(f'RAW {cohort} {cancer} complete; elapsed {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(rawrows).to_csv(out/'S34_RAW_HALLMARK.csv',index=False);pd.DataFrame(macrorows).to_csv(out/'S34_MACRO_MEAN_50.csv',index=False)
    t=data['TCGA'];pts=pd.read_csv(out/'P2_POINTS.csv').set_index(['cancer','target_id']);metrics=['matched_raw_r','composition_only_r','conditioned_r','paired_delta_r']
    for cancer in cfg['scope']['tcga_cancers']:
        ix=np.flatnonzero((t['cancer']==cancer)&t['eligible']);Y=t['Y'][ix];P=t['P'][ix];X=t['X'][ix];n=len(ix);outer=t['outer'][ix];inner=t['inner'][ix]
        splits=nuisance_splits(outer,inner);groups=[te for tr,te in splits];rng=rng_for('p2',cancer)
        def compute(a):
            for b in range(B):
                a[b]=p2_all(Y,P,X,splits,counts_for(groups,n,rng))
                if (b+1)%500==0:print(f'P2 {cancer} {b+1}/{B}; elapsed {time.time()-start:.1f}s',flush=True)
        v=save_or_compute('p2_'+cancer,(B,50,4),compute)
        for j,tar in enumerate(t['targets']):
            for k,metric in enumerate(metrics):
                lo,hi,nv,reason=interval(v[:,j,k]);p2rows.append({'cancer':cancer,'target_id':tar,'metric':metric,'n':n,'estimate':pts.loc[(cancer,tar),metric],'ci_low':lo,'ci_high':hi,'valid_resamples':nv,'attempted_resamples':B,'na_reason':reason,'interval_method':method+'; nuisance refitted in each replicate'})
        pd.DataFrame(p2rows).to_csv(out/'S35_COMPOSITION_HALLMARK.csv',index=False)
    for cancer in cfg['scope']['cptac_cancers']:
        rt=rawreps[('TCGA',cancer)];rc=rawreps[('CPTAC',cancer)];dz=fisher_z(rc)-fisher_z(rt)
        tt=pd.read_csv(out/'TCGA_POINTS.csv').set_index(['cancer','target_id']);cc=pd.read_csv(out/'CPTAC_POINTS.csv').set_index(['cancer','target_id'])
        for j,tar in enumerate(t['targets']):
            tr=tt.loc[(cancer,tar)];cr=cc.loc[(cancer,tar)];lo,hi,nv,reason=interval(dz[:,j])
            externalrows.append({'cancer':cancer,'target_id':tar,'n_tcga':int(tr.n),'n_cptac':int(cr.n),'tcga_r':tr.r,'cptac_r':cr.r,'delta_fisher_z':float(fisher_z(cr.r)-fisher_z(tr.r)),'ci_low':lo,'ci_high':hi,'valid_resamples':nv,'attempted_resamples':B,'na_reason':reason,'interval_method':method+'; independent cohort resampling; protocol-specific contrast'})
        p=repdir/('external_'+cancer+'.npz');np.savez_compressed(p,values=dz);p.with_suffix('.sha256').write_text(sha(p))
    pd.DataFrame(externalrows).to_csv(out/'S36_EXTERNAL_HALLMARK.csv',index=False)
    support=[]
    for c in cfg['scope']['tcga_cancers']:
        n=int(np.sum(t['cancer']==c));nn=int(np.sum((t['cancer']==c)&t['eligible']));support.append({'cancer':c,'tcga_n':n,'p2_n':nn,'p2_coverage_fraction':nn/n,'cptac_n':int(np.sum(data['CPTAC']['cancer']==c))})
    pd.DataFrame(support).to_csv(out/'SUPPORT.csv',index=False)
    for name,expected in [('S34_RAW_HALLMARK.csv',1200),('S34_MACRO_MEAN_50.csv',24),('S35_COMPOSITION_HALLMARK.csv',3200),('S36_EXTERNAL_HALLMARK.csv',400)]:
        df=pd.read_csv(out/name);assert len(df)==expected
        assert (df.attempted_resamples==2000).all() and (df.valid_resamples<=2000).all()
        assert (df.loc[df.valid_resamples<1900,['ci_low','ci_high']].isna().all().all())
    status={'status':'COMPUTATION_COMPLETE','seconds':time.time()-start,'attempts_per_branch_cancer':B,'no_new_p_values':True,'no_retraining':True,'point_gate':gate}
    (out/'RUN_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if __name__=='__main__':
    import argparse,os
    parser=argparse.ArgumentParser(description='S34–S36 fixed-predictor patient bootstrap')
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();os.chdir(args.root.resolve());main()
