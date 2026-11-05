"""M1A pilot-compatible per-slide tokens. No labels or cross-patient vocabulary."""
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from common import require,save_npz,record,write_json,verified,read_json,digest

TOKEN_POLICY={'k':100,'seed':2025,'n_init':1,'max_iter':300,'algorithm':'lloyd','short':'balanced_cyclic','sort':'original_kmeans_label_order'}

def tokens_100(x):
    a=np.asarray(x,dtype=np.float32);require(a.ndim==2 and len(a)>0 and np.isfinite(a).all(),'Invalid slide features')
    if len(a)<100:return a[np.arange(100)%len(a)],{'policy':TOKEN_POLICY,'n_patches':len(a),'replicated_tokens':100-len(a)}
    km=KMeans(n_clusters=100,random_state=2025,n_init=1,max_iter=300,algorithm='lloyd').fit(a)
    t=np.asarray(km.cluster_centers_,dtype=np.float32)
    require(t.shape==(100,a.shape[1]) and np.isfinite(t).all(),'Bad cluster centers')
    return t,{'policy':TOKEN_POLICY,'n_patches':len(a),'replicated_tokens':0,'occupied_clusters':len(np.unique(km.labels_))}

def make_patient(spec,out):
    """spec: patient_id,cancer,slides[{slide_id,feature record,feature_key}]."""
    import h5py
    require(len(spec['slides'])>0,'Empty slide roster')
    ids=[str(s['slide_id']) for s in spec['slides']];require(len(ids)==len(set(ids)),'Repeated slide')
    signature=digest({'spec':spec,'policy':TOKEN_POLICY});receipt=Path(str(out)+'.json')
    if receipt.exists():
        r=read_json(receipt);require(r['signature']==signature,'Existing token cache has different identity');verified(r['artifact']);return r
    require(not Path(out).exists(),'Unregistered token cache needs reconciliation')
    bags=[];counts=[];details=[]
    for s in spec['slides']:
        p=verified(s['feature'])
        with h5py.File(p,'r') as f:
            require(s['feature_key'] in f,'Missing explicit H5 feature key');a=np.asarray(f[s['feature_key']],dtype=np.float32)
        require(a.ndim==2 and a.shape[1]==1536,'Not UNI2H1536')
        t,d=tokens_100(a);bags.append(t);counts.append(len(a));details.append(d)
    save_npz(out,tokens=np.stack(bags),patch_counts=np.asarray(counts,dtype=np.int64),slide_id=np.asarray(ids,dtype=str),patient_id=np.asarray([spec['patient_id']],dtype=str))
    r={'signature':signature,'artifact':record(out),'details':details,'patient_id':spec['patient_id'],'cancer':spec['cancer'],'scope':'PER_SLIDE_FEATURE_ONLY_NO_LABELS'}
    write_json(receipt,r);return r

def patient_forward(model,tokens,patch_counts,microbatch_slides=4):
    import torch
    require(tokens.ndim==3 and len(tokens)==len(patch_counts)>0,'Invalid patient bag')
    require(bool(torch.isfinite(tokens).all()) and bool((patch_counts>0).all()),'Invalid bag values')
    chunks=[]
    for start in range(0,len(tokens),microbatch_slides): chunks.append(model(tokens[start:start+microbatch_slides]))
    pred=torch.cat(chunks,0);w=patch_counts.to(pred.dtype);w=w/w.sum()
    return (pred*w[:,None]).sum(0,keepdim=True)

def batch_patient_forward(model,patient_bags,microbatch_slides=4):
    """Microbatch slides, then weight within each patient. One supervised row/patient.
    Backpropagation retains the microbatch graphs; no stochastic patient dropping.
    """
    import torch
    require(len(patient_bags)>0,'Empty patient batch')
    lens=[len(x) for x,w in patient_bags]
    for x,w in patient_bags:
        require(x.ndim==3 and len(x)==len(w)>0,'Bad patient bag')
        require(bool(torch.isfinite(x).all()) and bool((w>0).all()),'Bad patient values')
    flat=torch.cat([x for x,w in patient_bags],0)
    pp=torch.cat([model(flat[i:i+microbatch_slides]) for i in range(0,len(flat),microbatch_slides)],0)
    out=[];start=0
    for (x,w),n in zip(patient_bags,lens):
        weight=w.to(pp.dtype);weight=weight/weight.sum()
        out.append((pp[start:start+n]*weight[:,None]).sum(0));start+=n
    return torch.stack(out)
