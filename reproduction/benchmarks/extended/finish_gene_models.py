"""Merge gene predictions, apply frozen FT01, and evaluate fixed functional targets.

External labels are read only by evaluate(), after prediction and scoring finish.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, json, os, sys, time
import numpy as np
from run_mosby_adapted import load, write, sha, ref

LABELS={'rnapath':'RNAPATH_UNI2H_GENE_ADAPTED',
        'path2omics':'PATH2OMICS_UNI2H_FFPE_GENE_ADAPTED'}

def csvwrite(p, rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def merge(root, cache, method):
    out=root/method; dest=out/'scoring_input';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        for r in load(dest/'COMPLETE.json')['outputs']:assert sha(r['path'])==r['sha256']
        return
    tc=load(cache/'TCGA16_index.json'); genes=load(root/'input/genes.json')['ids']; provenance=[]
    for fold in range(5):
        receipts=[]
        for cancer in tc['cancer_names']:
            p=out/f'F{fold}'/cancer/'COMPLETE.json';r=load(p)
            assert r['status']=='COMPLETE' and r['method']==method and r['fold']==fold and r['cancer']==cancer
            for x in r['outputs']:assert sha(x['path'])==x['sha256'],x['path']
            receipts.append(ref(p))
        q=out/f'F{fold}'/'COMPLETE.json'
        if not q.exists():write(q,{'status':'COMPLETE','method':method,'fold':fold,'outputs':receipts})
        provenance.append(ref(q))
    outputs=[];details=[]
    for cohort in ['TCGA16','CPTAC8']:
        idx=load(cache/f'{cohort}_index.json');ids=idx['patient_ids'];cs=idx['cancers'];by={x:i for i,x in enumerate(ids)}
        assert len(by)==len(ids)
        pred=np.zeros((len(ids),len(genes)),dtype=np.float64);seen=np.zeros(len(ids),dtype=np.int64)
        for fold in range(5):
            for cancer in tc['cancer_names']:
                p=out/f'F{fold}'/cancer/('TCGA_GENE.npz' if cohort=='TCGA16' else 'CPTAC_GENE.npz')
                if cohort=='CPTAC8' and cancer not in cs:
                    assert not p.exists();continue
                z=np.load(p);assert z['gene_ids'].astype(str).tolist()==genes
                zi=z['patient_ids'].astype(str).tolist();assert len(set(zi))==len(zi)
                ii=[by[x] for x in zi];assert z['cancers'].astype(str).tolist()==[cs[i] for i in ii]
                pp=z['prediction'].astype(np.float64);assert np.isfinite(pp).all()
                assert pp.shape==(len(ii),len(genes));pred[ii]+=pp;seen[ii]+=1
        expected=1 if cohort=='TCGA16' else 5
        assert (seen==expected).all(),(cohort,np.unique(seen,return_counts=True))
        pred/=expected
        np.save(dest/f'{cohort}_RAW_GENES.npy',pred)
        rec={'cohort':cohort,'n':len(ids),'genes':len(genes),'contributions_per_patient':expected,
             'negative_gene_fraction_before_projection':float((pred<0).mean()),
             'raw_gene_min':float(pred.min()),'raw_gene_max':float(pred.max()),
             'external_aggregation':'arithmetic mean of raw decoded gene predictions before max0 and FT01' if expected==5 else 'one held-out prediction per patient'}
        np.maximum(pred,0,out=pred);np.save(dest/f'{cohort}_PROJECTED.npy',pred)
        write(dest/f'{cohort}_AXES.json',{'patient_ids':ids,'cancers':cs,'gene_ids':genes,'target_ids':idx['target_ids']})
        details.append(rec)
        outputs += [ref(dest/f'{cohort}_{s}') for s in ['RAW_GENES.npy','PROJECTED.npy','AXES.json']]
    write(dest/'COMPLETE.json',{'status':'COMPLETE','method':method,'details':details,'fold_receipts':provenance,'outputs':outputs})
    print('GENE_MERGE_COMPLETE',method,json.dumps(details),flush=True)

SCORER=None; ARRAYS={}; AXES={}
def init_scorer(root, method):
    global SCORER,ARRAYS,AXES
    root=Path(root)
    from functional_scorer import FT01Scorer
    SCORER=FT01Scorer(root/'FT01_BINDING.json')
    for co in ['TCGA16','CPTAC8']:
        p=root/method/'scoring_input';ARRAYS[co]=np.load(p/f'{co}_PROJECTED.npy',mmap_mode='r');AXES[co]=load(p/f'{co}_AXES.json')

def score_chunk(task):
    root,method,co,start,end=task;root=Path(root)
    dest=root/method/'functional_predictions'/'chunks'/co;dest.mkdir(parents=True,exist_ok=True)
    p=dest/f'{start:05d}_{end:05d}.npz';receipt=p.with_suffix('.json')
    if receipt.exists():
        r=load(receipt);assert sha(p)==r['output']['sha256'];return r
    t=time.time();ax=AXES[co];ids=ax['patient_ids'][start:end]
    val=SCORER.score(ARRAYS[co][start:end],ids,ax['gene_ids'])
    assert val.shape==(len(ids),1564) and np.isfinite(val).all()
    np.savez(p,patient_ids=np.array(ids),target_ids=np.array(ax['target_ids']),prediction=val)
    r={'cohort':co,'start':start,'end':end,'seconds':time.time()-t,'output':ref(p),'binding_sha256':sha(root/'FT01_BINDING.json')}
    write(receipt,r);return r

def score(root,method,workers):
    out=root/method/'functional_predictions';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():
        for r in load(out/'COMPLETE.json')['outputs']:assert sha(r['path'])==r['sha256']
        return
    tasks=[]
    for co in ['TCGA16','CPTAC8']:
        n=len(load(root/method/'scoring_input'/f'{co}_AXES.json')['patient_ids'])
        tasks.extend((str(root),method,co,start,min(start+64,n)) for start in range(0,n,64))
    completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init_scorer,initargs=(str(root),method)) as pool:
        futures=[pool.submit(score_chunk,t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            r=f.result();completed.append(r)
            print('SCORED_CHUNK',method,len(completed),len(tasks),r['cohort'],r['start'],round(r['seconds'],2),flush=True)
    outputs=[]
    for co in ['TCGA16','CPTAC8']:
        ax=load(root/method/'scoring_input'/f'{co}_AXES.json');ids=ax['patient_ids'];pred=np.full((len(ids),1564),np.nan)
        for r in sorted((r for r in completed if r['cohort']==co),key=lambda r:r['start']):
            z=np.load(r['output']['path']);a,b=r['start'],r['end']
            assert z['patient_ids'].tolist()==ids[a:b] and z['target_ids'].tolist()==ax['target_ids'];pred[a:b]=z['prediction']
        assert np.isfinite(pred).all()
        p=out/f'{co}_MERGED.npz';np.savez(p,patient_ids=np.array(ids),cancers=np.array(ax['cancers']),target_ids=np.array(ax['target_ids']),prediction=pred);outputs.append(ref(p))
    write(out/'COMPLETE.json',{'status':'COMPLETE','method':method,'chunks':completed,'scorer_binding':ref(root/'FT01_BINDING.json'),'outputs':outputs})
    print('SCORING_COMPLETE',method,flush=True)

def evaluate(base,root,method):
    out=root/method;final=out/'evaluation';final.mkdir(exist_ok=True)
    if (final/'COMPLETE.json').exists():
        for r in load(final/'COMPLETE.json')['outputs']:assert sha(r['path'])==r['sha256']
        return
    fp=out/'functional_predictions'
    for r in load(fp/'COMPLETE.json')['outputs']:assert sha(r['path'])==r['sha256']
    metric_root=Path(__file__).resolve().parent;sys.path.insert(0,str(metric_root))
    from metrics import table,macro,coverage,KEYS
    catalog=load(base/'inputs/evaluation_sources.json')
    panelpath=base/'inputs/features/support_panel.json';panel=load(panelpath)
    overview=[];provenance=[ref(panelpath),ref(metric_root/'metrics.py'),ref(fp/'COMPLETE.json')]
    for co in ['TCGA16','CPTAC8']:
        z=np.load(fp/f'{co}_MERGED.npz');ids=z['patient_ids'].astype(str).tolist();cs=z['cancers'].astype(str).tolist();ts=z['target_ids'].astype(str).tolist()
        ent=catalog['sources'][co]['TRUTH'];src=ent['source'];assert sha(src['path'])==src['sha256'];provenance.append(src)
        a=np.load(src['path']);keys=ent['keys'];si=a[keys['patient_ids']].astype(str).tolist();st=a[keys['target_ids']].astype(str).tolist()
        assert set(si)==set(ids) and set(st)==set(ts) and len(set(si))==len(si)
        yi={x:i for i,x in enumerate(si)};ti={x:i for i,x in enumerate(st)}
        assert a[keys['cancers']][[yi[x] for x in ids]].astype(str).tolist()==cs
        y=a[keys['truth']][np.ix_([yi[x] for x in ids],[ti[x] for x in ts])]
        label=LABELS[method];rows=table(y,z['prediction'],cs,ts,panel,10);chosen=[r for r in rows if r['panel_membership']]
        summary={'cohort':co,'method':label,'patients_or_records':len(ids),'targets':len(ts),
                 'metrics':{k:macro(chosen,k) for k in KEYS},'coverage':coverage(chosen),'all_target_coverage':coverage(rows)}
        csvwrite(final/f'{co}_{label}_ALL_FUNCTIONS.csv',rows)
        pc=[]
        for c in sorted(set(cs)):
            cr=[r for r in chosen if r['cancer']==c]
            pc.append({'cohort':co,'method':label,'cancer':c,'n':cs.count(c),**{k:macro(cr,k)['value'] for k in KEYS}})
        csvwrite(final/f'{co}_{label}_PER_CANCER.csv',pc);write(final/f'{co}_{label}_SUMMARY.json',summary)
        overview.append({'cohort':co,'method':label,'n':len(ids),**{k:summary['metrics'][k]['value'] for k in KEYS},'status':summary['metrics']['pearson']['status']})
        print('EVALUATED',json.dumps(overview[-1]),flush=True)
    csvwrite(final/'OVERALL.csv',overview);write(final/'OVERALL.json',overview);write(final/'EVALUATION_PROVENANCE.json',provenance)
    write(final/'COMPLETE.json',{'status':'COMPLETE','new_method':LABELS[method],'no_new_CI_or_significance':True,'outputs':[ref(p) for p in final.iterdir() if p.is_file()]})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--method',choices=sorted(LABELS),required=True)
    a=p.parse_args();merge(a.root,a.cache,a.method);score(a.root,a.method,a.workers);evaluate(a.base,a.root,a.method)
