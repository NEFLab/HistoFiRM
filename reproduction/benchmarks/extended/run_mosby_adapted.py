"""MOSBY architecture adaptation on the fixed HistoFiRM bulk benchmark.

This is an independently implemented adaptation, not the authors' software.
Protocol and array identities are frozen before fitting. No external labels
are loaded by training or prediction.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def load(p):
    p=Path(p).resolve()
    def resolve(v):
        if isinstance(v,dict):
            out={k:resolve(x) for k,x in v.items()}
            if isinstance(v.get('path'),str):
                q=Path(v['path']);out['path']=str(q if q.is_absolute() else (p.parent/q).resolve())
            return out
        if isinstance(v,list):return [resolve(x) for x in v]
        return v
    return resolve(json.loads(p.read_text(encoding='utf-8')))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def ref(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def registry(base,co):return load(base/f'inputs/features/{co}_MORPHOLOGY_MMAP.json')

def prepare(base,out):
    import h5py
    out.mkdir(parents=True,exist_ok=False)
    protocol={'method':'MOSBY (UNI2-h adaptation)','reference':'10.1038/s41598-024-69198-6',
      'implementation':'independent architecture implementation; not author code',
      'inherited_architecture':'patch MLP 512/256; average pooling; 8000 tiles per slide cap',
      'adaptations':['UNI2-h 1536-dimensional features replace RetCCL',
        'fixed 1564 bulk functional targets replace original omic panels',
        'ReLU hidden layers and unconstrained linear output for signed normalized targets',
        'patient bag pools all retained tiles, including multiple slides',
        'training-only patient-balanced feature standardization and cancer-specific target standardization',
        'cancer-balanced masked MSE; same fixed fit/validation/outer/test partitions',
        'validation cancer-balanced Pearson selects epochs; no external target access',
        'refit outer-train for selected epoch count; CPTAC average of five decoded fold predictions'],
      'max_epochs':300,'patience':30,'learning_rate':0.001,'lr_step':5,'lr_gamma':0.9,
      'weight_decay':0.001,'batch_patients':64,'seed':2025,'hidden':[512,256],
      'dropout':0.0,'tile_cap_per_slide':8000,
      'cap_sampling':'fixed, without replacement, seed from slide identity and 2025; independent of labels',
      'selection_metric':'fixed-truth-support within-cancer Pearson, cancer equal weight',
      'evaluation_min_n':10,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(out/'PROTOCOL.json',protocol)
    sources=[]
    for co in ['TCGA16','CPTAC8']:
        rp=base/f'inputs/features/{co}_MORPHOLOGY_MMAP.json'
        reg=load(rp);sources.append(ref(rp));patients=reg['patients']
        counts=[sum(min(8000,s['n_patches']) for s in r['slides']) for r in patients]
        offset=np.r_[0,np.cumsum(counts)].astype(np.int64)
        data=np.lib.format.open_memmap(out/f'{co}_features.npy',mode='w+',dtype=np.float16,shape=(int(offset[-1]),1536))
        means=np.zeros((len(patients),1536),np.float64);seconds=np.zeros_like(means)
        def one(args):
            i,p=args;arrays=[];src=[]
            for s in p['slides']:
                q=s['features']['source_ref'];path=Path(q['path'])
                assert path.stat().st_size==q['bytes'],str(path)
                h=sha(path);assert h==q['sha256'],'H5_SHA_MISMATCH '+str(path)
                with h5py.File(path,'r') as f:a=f[q['key']][:]
                assert a.shape==(s['n_patches'],1536) and np.isfinite(a).all()
                if len(a)>8000:
                    seed=int(hashlib.sha256((s['slide_id']+'|2025').encode()).hexdigest()[:16],16)
                    ix=np.sort(np.random.default_rng(seed).choice(len(a),8000,replace=False));a=a[ix]
                arrays.append(a);src.append({'path':str(path),'sha256':h,'tiles_used':len(a),'tiles_available':s['n_patches']})
            a=np.concatenate(arrays);data[offset[i]:offset[i+1]]=a
            d=a.astype(np.float64);means[i]=d.mean(0);seconds[i]=(d*d).mean(0)
            return i,src
        allsrc=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for n,(i,src) in enumerate(pool.map(one,enumerate(patients))):
                allsrc.extend(src)
                if n%500==0:print(json.dumps({'prepare':co,'patients_done':n+1}),flush=True)
        data.flush();del data
        np.savez(out/f'{co}_moments.npz',mean=means,second=seconds)
        idx={'patient_ids':[p['patient_id'] for p in patients], 'cancers':[p['cancer'] for p in patients],
             'target_ids':reg['target_ids'],'cancer_names':reg['cancer_names'],'offsets':offset.tolist()}
        assert len(set(idx['patient_ids']))==len(patients)
        write(out/f'{co}_index.json',idx);write(out/f'{co}_H5_VERIFIED.json',allsrc)
    # Freeze exact registered splits, training label identities, and leakage checks.
    parts={};alltests=[]
    for f in range(5):
        parts[f]={}
        for part in ['fit','validation','outer_train','test']:
            p=base/f'inputs/folds/F{f}/{part}.json'
            r=load(p);ids=[x['patient_id'] for x in r['patients']]
            parts[f][part]=ids;sources.append(ref(p))
            if part!='test':
                lp=Path(r['labels']['path']);assert sha(lp)==r['labels']['sha256'];sources.append(ref(lp))
        a=parts[f]
        assert not set(a['fit'])&set(a['validation'])
        assert set(a['fit'])|set(a['validation'])==set(a['outer_train'])
        assert not set(a['outer_train'])&set(a['test'])
        alltests+=a['test']
    tc=load(out/'TCGA16_index.json');cp=load(out/'CPTAC8_index.json')
    assert len(alltests)==len(set(alltests))==6781 and set(alltests)==set(tc['patient_ids'])
    assert not set(tc['patient_ids'])&set(cp['patient_ids'])
    write(out/'SPLITS.json',parts);write(out/'INPUT_PROVENANCE.json',sources)
    write(out/'PREPARED.json',{'status':'COMPLETE','TCGA_patients':6781,'CPTAC_records':1128,
          'outputs':[ref(p) for p in out.glob('*') if p.is_file() and p.name!='PREPARED.json']})
    print('PREPARED_COMPLETE',flush=True)

def train(base,out,fold,smoke=False):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    assert (out/'PREPARED.json').exists()
    cfg=load(out/'PROTOCOL.json');splits=load(out/'SPLITS.json')[str(fold)]
    idx=load(out/'TCGA16_index.json');names=idx['cancer_names'];cs=np.array(idx['cancers'])
    by={x:i for i,x in enumerate(idx['patient_ids'])};cidx=np.array([names.index(c) for c in cs])
    offsets=np.array(idx['offsets']);raw=np.load(out/'TCGA16_features.npy',mmap_mode='r')
    mom=np.load(out/'TCGA16_moments.npz');dest=out/('smoke' if smoke else f'F{fold}');dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():raise RuntimeError('Already completed')
    # Copy immutable TCGA patch bank to GPU; normalization remains fit-specific.
    bank=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
    for start in range(0,len(raw),32000):bank[start:start+32000]=torch.from_numpy(raw[start:start+32000].copy()).cuda()
    class Net(nn.Module):
        def __init__(self,mu,scale):
            super().__init__();self.register_buffer('mu',torch.tensor(mu,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
            self.layers=nn.Sequential(nn.Linear(1536,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
            self.head=nn.Linear(256,1564)
        def forward(self,features,lengths):
            z=self.layers((features.float()-self.mu)/self.scale)
            # Linear output commutes with mean pooling; avoid materializing tiles x 1564.
            pooled=torch.segment_reduce(z.float(),'mean',lengths=lengths)
            return self.head(pooled)
    def batch(rows,source=None,off=offsets):
        if source is None:source=bank
        lens=[int(off[i+1]-off[i]) for i in rows]
        x=torch.cat([source[off[i]:off[i+1]] for i in rows],dim=0)
        return x,torch.tensor(lens,device='cuda')
    def setup(part):
        r=load(base/f'inputs/folds/F{fold}/{part}.json')
        a=np.load(r['labels']['path'],allow_pickle=False)
        assert a['patient_ids'].tolist()==splits[part] and a['target_ids'].tolist()==idx['target_ids']
        ids=np.array([by[x] for x in splits[part]]);y=a['truth'];yc=a['cancers'].astype(str)
        means=np.zeros((len(names),1564));scales=np.ones_like(means);valid=np.zeros_like(means,dtype=bool);support=np.zeros_like(valid)
        for j,c in enumerate(names):
            yy=y[yc==c];finite=np.isfinite(yy);n=finite.sum(0);m=np.divide(np.where(finite,yy,0).sum(0),n,out=np.zeros(1564),where=n>0)
            ss=np.where(finite,yy-m,0);var=(ss*ss).sum(0);sd=np.sqrt(np.divide(var,n-1,out=np.zeros(1564),where=n>1))
            means[j]=m;scales[j]=np.where(sd>1e-10,sd,1);valid[j]=(n>=2)&(var>1e-20);support[j]=n>=2
        cc=cidx[ids];mask=np.isfinite(y)&valid[cc]
        z=np.where(mask,(y-means[cc])/scales[cc],0).astype(np.float32)
        counts=np.bincount(cc,minlength=len(names));weights=len(ids)/(len(names)*counts[cc])
        mu=mom['mean'][ids].mean(0);sd=np.sqrt(np.maximum(mom['second'][ids].mean(0)-mu*mu,0));sd=np.where(sd>1e-6,sd,1)
        random.seed(2025+fold);np.random.seed(2025+fold);torch.manual_seed(2025+fold);torch.cuda.manual_seed_all(2025+fold)
        model=Net(mu,sd).cuda();optim=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.StepLR(optim,step_size=5,gamma=0.9)
        return ids,z,mask,weights,model,optim,sched,means,scales,valid,support
    @torch.inference_mode()
    def predict(model,rows,source=None,off=offsets):
        model.eval();p=[]
        for st in range(0,len(rows),32):
            x,l=batch(rows[st:st+32],source,off)
            with torch.autocast('cuda',dtype=torch.bfloat16):v=model(x,l)
            p.append(v.float().cpu().numpy())
        return np.concatenate(p)
    def epoch_train(pack,epoch):
        ids,z,mask,weights,model,optim,sched,*_=pack;model.train();losses=[]
        rng=np.random.default_rng(2025+fold*10000+epoch);order=rng.permutation(len(ids))
        for start in range(0,len(ids),64):
            pos=order[start:start+64];optim.zero_grad(set_to_none=True)
            x,l=batch(ids[pos]);yy=torch.tensor(z[pos],device='cuda');mm=torch.tensor(mask[pos],device='cuda');ww=torch.tensor(weights[pos],dtype=torch.float32,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,l)
            loss=((((p.float()-yy)**2)*mm).sum(1)/mm.sum(1).clamp_min(1)*ww).mean()
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
            loss.backward();optim.step();losses.append(float(loss.detach()))
        sched.step();return float(np.mean(losses))
    sys.path.insert(0,str(Path(__file__).parent))
    from statistics_fixed import selection
    pack=setup('fit');val=np.load(load(base/f'inputs/folds/F{fold}/validation.json')['labels']['path'])
    valrows=np.array([by[x] for x in val['patient_ids']]);best=-np.inf;best_epoch=0
    if smoke:
        a=list(pack);a[:4]=[v[:128] for v in a[:4]];pack=tuple(a)
    for epoch in range(1,(1 if smoke else cfg['max_epochs'])+1):
        t=time.time();loss=epoch_train(pack,epoch);p=predict(pack[4],valrows)
        cc=cidx[valrows];decoded=p*pack[8][cc]+pack[7][cc]
        sc=selection(val['truth'][:len(valrows)],decoded,val['cancers'][:len(valrows)],{c:pack[9][j] for j,c in enumerate(names)})
        score=sc['pearson']
        if score>best+1e-12:best=score;best_epoch=epoch
        row={'phase':'select','fold':fold,'epoch':epoch,'loss':loss,'val_pearson':score,'best_epoch':best_epoch,'seconds':time.time()-t}
        with open(dest/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    write(dest/'SELECTION.json',{'epochs':best_epoch,'validation_pearson':best,'protocol_sha256':sha(out/'PROTOCOL.json')})
    if smoke:return
    del pack;torch.cuda.empty_cache();pack=setup('outer_train')
    for epoch in range(1,best_epoch+1):
        t=time.time();loss=epoch_train(pack,epoch)
        row={'phase':'refit','fold':fold,'epoch':epoch,'epochs':best_epoch,'loss':loss,'seconds':time.time()-t}
        with open(dest/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    model=pack[4];model.eval();means,scales,valid,support=pack[7:11]
    torch.save({'state_dict':model.state_dict(),'target_center':means,'target_scale':scales,'target_support':support,'epochs':best_epoch,'protocol':cfg},dest/'model.pt')
    def export(co,ids,cancers,rows,source,off):
        p=predict(model,rows,source,off);ci=np.array([names.index(c) for c in cancers])
        y=p*scales[ci]+means[ci];y=np.where(valid[ci],y,means[ci]);y=np.where(support[ci],y,np.nan)
        np.savez(dest/f'{co}_prediction.npz',patient_ids=np.array(ids),cancers=np.array(cancers),target_ids=np.array(idx['target_ids']),prediction=y)
    rows=np.array([by[x] for x in splits['test']]);export('TCGA16',splits['test'],cs[rows].tolist(),rows,bank,offsets)
    # Release training patch storage before loading external images; labels remain unopened.
    del bank;torch.cuda.empty_cache()
    ci=load(out/'CPTAC8_index.json');ext=np.load(out/'CPTAC8_features.npy',mmap_mode='r')
    eb=torch.empty(ext.shape,dtype=torch.float16,device='cuda')
    for st in range(0,len(ext),32000):eb[st:st+32000]=torch.from_numpy(ext[st:st+32000].copy()).cuda()
    export('CPTAC8',ci['patient_ids'],ci['cancers'],np.arange(len(ci['patient_ids'])),eb,np.array(ci['offsets']))
    write(dest/'COMPLETE.json',{'status':'COMPLETE','fold':fold,'selected_epochs':best_epoch,'outputs':[ref(dest/n) for n in ['model.pt','TCGA16_prediction.npz','CPTAC8_prediction.npz','training.jsonl','SELECTION.json']]})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train','smoke']);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args()
    if a.mode=='prepare':prepare(a.base,a.out)
    else:train(a.base,a.out,a.fold,a.mode=='smoke')
