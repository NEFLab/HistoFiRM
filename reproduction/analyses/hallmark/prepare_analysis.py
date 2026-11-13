"""Verify aligned patient arrays against saved estimates before fixed-model bootstrap."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
from hallmark_core import corr_cols,nuisance_splits,p2_all,verify_reference
from profile_core import patient_pearson,summary_schema,summaries

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def validate(d,co,profiles):
    required={'ids','cancer','targets','Y','P'}|({'outer'} if co=='TCGA' or profiles else set())
    assert required<=set(d),'Missing input arrays'
    n=len(d['ids']);assert len(set(d['ids']))==n
    assert len(d['targets'])==len(set(d['targets']))==50
    assert all(str(t).startswith('HALLMARK::') for t in d['targets'])
    assert len(d['cancer'])==n and d['Y'].shape==(n,50)
    assert d['P'].shape==((n,7,50) if profiles else (n,50))
    assert np.isfinite(d['Y']).all() and np.isfinite(d['P']).all()
    assert len(set(d['cancer']))==(16 if co=='TCGA' else 8)
    if co=='TCGA':assert set(d['outer'])==set(range(5)) and len(d['outer'])==n
    elif profiles:assert np.array_equal(d['outer'],np.zeros(n,dtype=int))

def bind(a):
    root=a.output.resolve()
    if root.exists():raise ValueError('Use a new output directory.')
    data={'TCGA':read_npz(a.tcga),'CPTAC':read_npz(a.cptac)}
    for co,d in data.items():validate(d,co,a.mode=='profiles')
    assert np.array_equal(data['TCGA']['targets'],data['CPTAC']['targets'])
    assert not set(data['TCGA']['ids'])&set(data['CPTAC']['ids'])
    root.mkdir(parents=True);inp=root/'private_inputs';inp.mkdir();out=root/'outputs';out.mkdir()
    for co,d in data.items():np.savez_compressed(inp/(co+'.npz'),**d)
    if a.mode=='hallmark':
        assert a.s34 is not None and a.s35 is not None
        rawref=pd.read_csv(a.s34);compref=pd.read_csv(a.s35)
        cfg={'statistics':{'bootstrap_replicates':2000},'scope':{co.lower()+'_cancers':sorted(set(d['cancer'].tolist())) for co,d in data.items()}}
        dump(root/'config.json',cfg);gates={}
        for co,d in data.items():
            rows=[]
            for c in cfg['scope'][co.lower()+'_cancers']:
                ix=np.flatnonzero(d['cancer']==c);rs=corr_cols(d['Y'][ix],d['P'][ix],np.ones(len(ix)))
                rows.extend({'cancer':c,'target_id':t,'r':float(r),'n':len(ix)} for t,r in zip(d['targets'],rs))
            frame=pd.DataFrame(rows);ref=rawref[rawref.cohort==co].rename(columns={'estimate':'r'})
            gates[co]=verify_reference(frame,ref,['r'],out/(co+'_POINT_AUDIT.csv'))
            frame.to_csv(out/(co+'_POINTS.csv'),index=False)
        d=data['TCGA'];n=len(d['ids'])
        assert {'eligible','X','inner'}<=set(d)
        assert d['eligible'].dtype==bool and d['eligible'].shape==(n,) and d['X'].shape==(n,2)
        assert int(d['eligible'].sum())==5646 and len(d['inner'])==n
        metrics=['matched_raw_r','composition_only_r','conditioned_r','paired_delta_r'];rows=[]
        for c in cfg['scope']['tcga_cancers']:
            ix=np.flatnonzero((d['cancer']==c)&d['eligible'])
            assert np.isfinite(d['X'][ix]).all()
            splits=nuisance_splits(d['outer'][ix],d['inner'][ix])
            values=p2_all(d['Y'][ix],d['P'][ix],d['X'][ix],splits,np.ones(len(ix),int))
            for j,t in enumerate(d['targets']):rows.append({'cancer':c,'target_id':t,'n':len(ix),**dict(zip(metrics,values[j]))})
        frame=pd.DataFrame(rows)
        ref=compref.pivot(index=['cancer','target_id','n'],columns='metric',values='estimate').reset_index()
        gates['composition']=verify_reference(frame,ref,metrics,out/'P2_POINT_AUDIT.csv')
        frame.to_csv(out/'P2_POINTS.csv',index=False)
        refs=[a.s34,a.s35]
    else:
        assert a.target_reference is not None and a.summary_reference is not None
        cfg=json.loads((HERE/'profile_config.json').read_text());models=cfg['models']
        names=dict(zip(models,['HistoFiRM','Same-split Ridge','Gated ABMIL','MOSBY','TIGER','Path2Omics','RNAPath']))
        ref=pd.read_csv(a.target_reference);sref=pd.read_csv(a.summary_reference)
        raw={};cancers={};audit=[]
        for co,d in data.items():
            assert 'models' in d and d['models'].tolist()==models
            assert d['targets'].tolist()==sorted(d['targets'].tolist())
            cancers[co]=sorted(set(d['cancer'].tolist()));points=[]
            for c in cancers[co]:
                ix=np.flatnonzero(d['cancer']==c);v=patient_pearson(d['Y'][ix],d['P'][ix],np.ones((1,len(ix))))[0]
                assert np.isfinite(v).all() and (np.abs(v)<1).all();points.append(v)
                for mi,m in enumerate(models):
                    rr=ref[(ref.cohort==co)&(ref.cancer==c)&(ref.model_id==m)].set_index('target_id').loc[d['targets']]
                    assert (rr.n_total==len(ix)).all()
                    for ti,t in enumerate(d['targets']):audit.append({'cohort':co,'cancer':c,'model_id':m,'target_id':t,'expected':float(rr.loc[t,'pearson']),'reproduced':float(v[mi,ti]),'abs_difference':float(abs(v[mi,ti]-rr.loc[t,'pearson']))})
            raw[co]=np.stack(points)[None,:,:,:]
        points=pd.DataFrame(audit);assert len(points)==8400 and points.abs_difference.max()<cfg['point_tolerance']
        points.to_csv(out/'PATIENT_POINT_REPRODUCTION.csv',index=False)
        common=sorted(set(cancers['TCGA'])&set(cancers['CPTAC']));assert len(common)==8
        schema=summary_schema(models,names);values=summaries(raw,cancers,common)[0]
        assert sref.summary_id.is_unique and len(sref)==38;sr=sref.set_index('summary_id')
        for row,v in zip(schema,values):
            expected=float(sr.loc[row['summary_id'],'estimate']);assert abs(v-expected)<cfg['point_tolerance']
            row.update(estimate=float(v),original_A_estimate=expected,abs_difference=float(abs(v-expected)))
        pd.DataFrame(schema).to_csv(out/'SUMMARY_POINT_REPRODUCTION.csv',index=False)
        dump(root/'config.json',cfg);dump(root/'AXES.json',{'cancers':cancers,'shared_cancers':common,'targets':data['TCGA']['targets'].tolist(),'models':models,'names':names})
        gates={'target_rows':8400,'summary_rows':38,'target_max_abs_error':float(points.abs_difference.max())};refs=[a.target_reference,a.summary_reference]
    # Saved references and numerical source files are pinned before any bootstrap.
    rd=root/'references';rd.mkdir()
    for i,p in enumerate(refs):shutil.copy2(p,rd/(str(i)+'_'+p.name))
    paths=list(inp.iterdir())+list(rd.iterdir())+list(out.iterdir())+[root/'config.json']+list(HERE.glob('*.py'))
    if (root/'AXES.json').exists():paths.append(root/'AXES.json')
    manifest=[{'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
    dump(root/'INPUT_MANIFEST.json',manifest)
    dump(root/'POINT_GATE.json',{'status':'PASS','gates':gates,'manifest_sha256':sha(root/'INPUT_MANIFEST.json')})
    print(json.dumps({'status':'PASS','mode':a.mode,'output':str(root),'bootstrap_started':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['hallmark','profiles']);p.add_argument('--tcga',type=Path,required=True);p.add_argument('--cptac',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--s34',type=Path);p.add_argument('--s35',type=Path);p.add_argument('--target-reference',type=Path);p.add_argument('--summary-reference',type=Path)
    bind(p.parse_args())
