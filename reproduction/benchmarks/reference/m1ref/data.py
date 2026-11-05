"""Only explicit training arrays or feature-only external arrays enter these APIs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import h5py
from .contracts import *

META_FIELDS={'patient_id','cancer','cohort','outer_fold'}

def metadata(path, cohort):
    frame=pd.read_csv(path,dtype={'patient_id':str,'cancer':str,'cohort':str})
    require(set(frame.columns)==META_FIELDS,'Metadata must not contain RNA/clinical/extra columns')
    unique(frame.patient_id)
    require(set(frame.cohort)=={cohort},f'Expected only cohort {cohort}')
    if cohort=='TCGA':
        require(frame.outer_fold.isin(range(5)).all(),'Use original P07 fold labels 0..4')
    return frame

def load_dataset(path, kind):
    cfg=read_json(path)
    allowed={'kind','metadata','X','target_ids','Y','input_receipt','fold_lineage','feature_dim','gene_state'}
    require(set(cfg).issubset(allowed),'Unknown dataset key (possibly external truth)')
    require(cfg['kind']==kind,'Dataset role mismatch')
    if kind=='EXTERNAL_FEATURES':
        require('Y' not in cfg and 'target_ids' not in cfg and 'gene_state' not in cfg,'External inference must be label free')
        m=metadata(verify_record(cfg['metadata']),'CPTAC')
    else:
        require(kind in ('TCGA_F','TCGA_G'),'Training kind must be TCGA_F or TCGA_G')
        m=metadata(verify_record(cfg['metadata']),'TCGA')
    x=np.load(verify_record(cfg['X']),mmap_mode='r',allow_pickle=False)
    require(x.ndim==2 and x.shape==(len(m),int(cfg['feature_dim'])),'Feature dimensions/order not bound')
    require(np.isfinite(x).all(),'Nonfinite mean features')
    require(cfg.get('input_receipt') is not None,'Input aggregation/identity receipt required')
    inp=read_json(verify_record(cfg['input_receipt']))
    require(inp['patient_order_digest']==digest(m.patient_id.tolist()),'Feature row order receipt mismatch')
    y=targets=None
    if kind!='EXTERNAL_FEATURES':
        y=np.load(verify_record(cfg['Y']),mmap_mode='r',allow_pickle=False)
        targets=unique(read_json(verify_record(cfg['target_ids']))['ids'],'targets')
        require(y.shape==(len(m),len(targets)),'Training label shape mismatch')
        require(not np.isinf(y).any(),'Inf labels')
        require(not np.any(np.all(~np.isfinite(y),axis=1)),'Patient has no training target values')
        require(cfg.get('fold_lineage') is not None,'Existing P07 fold-source required')
        lineage=read_json(verify_record(cfg['fold_lineage']))
        require(lineage.get('patient_fold_digest')==digest(m[['patient_id','outer_fold']].values.tolist()),'P07 fold lineage mismatch')
        if kind=='TCGA_G':
            require(cfg.get('gene_state')=='HGNC_ORDERED_LOG2_TPM_PLUS1','Gene output space is fixed log2(TPM+1)')
            require(np.isfinite(y).all() and np.min(y)>=0,'All fixed background genes required; no missing/negative log2 TPM')
    return cfg,m,x,y,targets

def inner_indices(meta, outer_fold, seed=2025, fraction=.2):
    """One fixed cancer-stratified inner holdout, NEVER derives labels from outcomes."""
    train=[];valid=[]
    for c in sorted(meta.cancer.unique()):
        ids=np.flatnonzero((meta.cancer.values==c)&(meta.outer_fold.values!=outer_fold))
        require(len(ids)>=5,'Too few development patients in cancer')
        ordered=sorted(ids,key=lambda i:hashlib.sha256(f'{seed}|{outer_fold}|{meta.iloc[i].patient_id}'.encode()).hexdigest())
        nv=max(1,int(np.floor(len(ordered)*fraction)))
        valid.extend(ordered[:nv]);train.extend(ordered[nv:])
    test=np.flatnonzero(meta.outer_fold.values==outer_fold)
    disjoint(meta.iloc[train].patient_id,meta.iloc[valid].patient_id,meta.iloc[test].patient_id)
    return np.asarray(sorted(train)),np.asarray(sorted(valid)),test

def stream_patient_mean(assets, input_dim=1536, chunk=8192):
    """All patches, patch-count-weighted across registered slides. No per-slide equal mean."""
    sums=np.zeros(input_dim,dtype=np.float64);n=0;seen=set()
    for item in assets:
        p=verify_record(item);key=item['dataset'];ident=(str(p.resolve()),key)
        require(ident not in seen,'Duplicate feature asset');seen.add(ident)
        with h5py.File(p,'r') as f:
            require(key in f,'Explicit feature dataset missing');d=f[key]
            require(d.ndim==2 and d.shape[1]==input_dim and d.shape[0]>0,'Bad/empty patch matrix')
            for start in range(0,len(d),chunk):
                b=np.asarray(d[start:start+chunk],dtype=np.float64)
                require(np.isfinite(b).all(),'Nonfinite patch features')
                sums+=b.sum(0);n+=len(b)
    require(n>0,'No patches');return sums/n,n

def prepare_mean_cache(manifest_path,out_dir):
    """IO binding is explicit, not a filename/glob search. Resume requires signature equality."""
    import tempfile,os
    man=read_json(manifest_path); rows=man['patients'];ids=unique([r['patient_id'] for r in rows])
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    sig=digest(man);done=out/'INPUT_RECEIPT.json'
    if done.exists():
        r=read_json(done);require(r['input_signature']==sig,'Cache belongs to another asset mapping')
        verify_record(r['X']);return r
    target=out/'X.npy';require(not target.exists(),'Unregistered final cache exists; reconcile, do not overwrite')
    tmp=out/'X.partial.npy'
    require(not tmp.exists(),'Interrupted cache: inspect before rebuilding, do not hide failure')
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype='float64',shape=(len(rows),man['feature_dim']))
    counts=[]
    for i,row in enumerate(rows):
        arr[i],n=stream_patient_mean(row['assets'],man['feature_dim']);counts.append(n)
    arr.flush();del arr
    os.link(tmp,target);tmp.unlink()
    rec={'input_signature':sig,'patient_order_digest':digest(ids.tolist()),'n_patients':len(ids),
         'feature_dim':man['feature_dim'],'aggregation':'ALL_VALID_PATCH_MEAN_FLOAT64','patch_counts':counts,'X':record(target)}
    write_json(done,rec);return rec
