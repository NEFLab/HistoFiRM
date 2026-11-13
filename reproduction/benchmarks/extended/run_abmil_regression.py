"""Gated-ABMIL multioutput regression control on fixed UNI2-h inputs.
Not a reproduction of the complete CAMIL or TME-WSI pipelines.
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

def train(base,out,fold,smoke=False):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    assert (out/'PREPARED.json').exists()
    cfg=load(out/'PROTOCOL.json');cache=Path(cfg['cache']);splits=load(cache/'SPLITS.json')[str(fold)]
    idx=load(cache/'TCGA16_index.json');names=idx['cancer_names'];cs=np.array(idx['cancers'])
    by={x:i for i,x in enumerate(idx['patient_ids'])};cidx=np.array([names.index(c) for c in cs])
    offsets=np.array(idx['offsets']);raw=np.load(cache/'TCGA16_features.npy',mmap_mode='r')
    mom=np.load(cache/'TCGA16_moments.npz');dest=out/('smoke' if smoke else f'F{fold}');dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():raise RuntimeError('Already completed')
    # Copy immutable TCGA patch bank to GPU; normalization remains fit-specific.
    bank=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
    for start in range(0,len(raw),32000):bank[start:start+32000]=torch.from_numpy(raw[start:start+32000].copy()).cuda()
    class Net(nn.Module):
        def __init__(self,mu,scale):
            super().__init__();self.register_buffer('mu',torch.tensor(mu,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
            self.encoder=nn.Sequential(nn.Linear(1536,512),nn.ReLU(),nn.Dropout(.25))
            self.v=nn.Sequential(nn.Linear(512,256),nn.Tanh(),nn.Dropout(.25))
            self.u=nn.Sequential(nn.Linear(512,256),nn.Sigmoid(),nn.Dropout(.25))
            self.att=nn.Linear(256,1);self.head=nn.Linear(512,1564)
        def forward(self,features,lengths):
            z=self.encoder((features.float()-self.mu)/self.scale)
            score=self.att(self.v(z)*self.u(z)).float()
            maximum=torch.segment_reduce(score,'max',lengths=lengths)
            unnorm=torch.exp(score-torch.repeat_interleave(maximum,lengths,dim=0))
            denom=torch.segment_reduce(unnorm,'sum',lengths=lengths)
            weight=unnorm/torch.repeat_interleave(denom,lengths,dim=0)
            pooled=torch.segment_reduce(weight*z.float(),'sum',lengths=lengths)
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
        model=Net(mu,sd).cuda();optim=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=lambda _:1.)
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
        vm=np.isfinite(val['truth'])&pack[9][cc]
        vz=np.where(vm,(val['truth']-pack[7][cc])/pack[8][cc],0)
        vw=len(cc)/(len(names)*np.bincount(cc,minlength=len(names))[cc])
        score=-float(np.mean((((p-vz)**2)*vm).sum(1)/np.maximum(vm.sum(1),1)*vw))
        if score>best+1e-12:best=score;best_epoch=epoch
        row={'phase':'select','fold':fold,'epoch':epoch,'loss':loss,'validation_negative_mse':score,'best_epoch':best_epoch,'seconds':time.time()-t}
        with open(dest/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if epoch>=cfg['min_epochs'] and epoch-best_epoch>=cfg['patience']:break
    write(dest/'SELECTION.json',{'epochs':best_epoch,'validation_negative_mse':best,'protocol_sha256':sha(out/'PROTOCOL.json')})
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
    ci=load(cache/'CPTAC8_index.json');ext=np.load(cache/'CPTAC8_features.npy',mmap_mode='r')
    eb=torch.empty(ext.shape,dtype=torch.float16,device='cuda')
    for st in range(0,len(ext),32000):eb[st:st+32000]=torch.from_numpy(ext[st:st+32000].copy()).cuda()
    export('CPTAC8',ci['patient_ids'],ci['cancers'],np.arange(len(ci['patient_ids'])),eb,np.array(ci['offsets']))
    write(dest/'COMPLETE.json',{'status':'COMPLETE','fold':fold,'selected_epochs':best_epoch,'outputs':[ref(dest/n) for n in ['model.pt','TCGA16_prediction.npz','CPTAC8_prediction.npz','training.jsonl','SELECTION.json']]})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','smoke']);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args()
    train(a.base,a.out,a.fold,a.mode=='smoke')
