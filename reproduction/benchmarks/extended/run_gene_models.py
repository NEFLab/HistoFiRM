"""Patient-level UNI2-h adaptations of RNAPath and the Path2Omics FFPE branch.

Train on bulk log2(TPM+1), then apply the separately frozen functional scorer.
All paths are arguments. External RNA labels are never read here.
"""
from pathlib import Path
import argparse, hashlib, json, time, random, gc
import numpy as np
import torch
from torch import nn

def load(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def ref(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))

class RNAPath(nn.Module):
    def __init__(self,dim,genes,mu,sd,bias=None):
        super().__init__();self.register_buffer('mu',torch.as_tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.as_tensor(sd,dtype=torch.float32))
        self.drop=nn.Dropout(.1);self.linear=nn.Linear(dim,genes)
        nn.init.xavier_normal_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,x,lengths):
        h=self.drop((x.float()-self.mu)/self.sd)
        # Activation precedes aggregation, preserving the RNAPath architecture.
        y=torch.relu(self.linear(h))
        return torch.segment_reduce(y.float(),'mean',lengths=lengths)

class Path2Omics(nn.Module):
    def __init__(self,dim,genes,mu,sd,bias=None):
        super().__init__();self.register_buffer('mu',torch.as_tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.as_tensor(sd,dtype=torch.float32))
        self.first=nn.Linear(dim,512);self.drop=nn.Dropout(.2);self.last=nn.Linear(512,genes)
        if bias is not None:
            with torch.no_grad():self.last.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
    def forward(self,x,lengths):
        h=self.drop(self.first((x.float()-self.mu)/self.sd))
        # Mean and the final affine layer commute; retain tile-level dropout.
        return self.last(torch.segment_reduce(h.float(),'mean',lengths=lengths))

def genecorr(y,p):
    yy=y.astype(np.float64)-y.mean(0,dtype=np.float64);pp=p.astype(np.float64)-p.mean(0,dtype=np.float64)
    den=np.sqrt((yy*yy).sum(0)*(pp*pp).sum(0));ok=den>1e-20
    if not ok.any():return -1.
    return float(np.mean((yy*pp).sum(0)[ok]/den[ok]))

def main(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    cfg=load(a.root/'PROTOCOL.json')[a.method];cache=a.cache;idx=load(cache/'TCGA16_index.json');ext=load(cache/'CPTAC8_index.json')
    splits=load(cache/'SPLITS.json')[str(a.fold)]
    genes=load(a.root/'input/genes.json')['ids'];gene_y=np.load(a.root/'input/TCGA_GENE_TRUTH.npy',mmap_mode='r')
    assert load(a.root/'input/patient_ids.json')==idx['patient_ids']
    mapping={p:i for i,p in enumerate(idx['patient_ids'])};cancers=np.array(idx['cancers']);offset=np.array(idx['offsets'])
    mom=np.load(cache/'TCGA16_moments.npz');features=np.load(cache/'TCGA16_features.npy',mmap_mode='r')
    extfeat=np.load(cache/'CPTAC8_features.npy',mmap_mode='r');extoff=np.array(ext['offsets'])
    out=a.root/a.method/('smoke' if a.smoke else f'F{a.fold}');out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Completed fold must not be overwritten')
    classes={'rnapath':RNAPath,'path2omics':Path2Omics}
    selected=[a.cancer] if a.cancer else idx['cancer_names']
    for cancer in selected:
        dest=out/cancer;dest.mkdir(exist_ok=True)
        if (dest/'COMPLETE.json').exists():continue
        rows=np.where(cancers==cancer)[0];by={int(v):i for i,v in enumerate(rows)}
        lengths=np.array([offset[i+1]-offset[i] for i in rows]);off=np.r_[0,lengths.cumsum()]
        bank=torch.empty((int(off[-1]),1536),dtype=torch.float16,device='cuda')
        for j,i in enumerate(rows):bank[off[j]:off[j+1]]=torch.from_numpy(features[offset[i]:offset[i+1]].copy()).cuda()
        def batch(rr,rng=None):
            chunks=[];lens=[]
            for i in rr:
                j=by[int(i)];x=bank[off[j]:off[j+1]]
                if rng is not None and a.method=='rnapath':
                    n=max(1,int(len(x)*rng.uniform(.7,1.)));sel=torch.as_tensor(rng.choice(len(x),n,replace=False),device='cuda');x=x[sel]
                chunks.append(x);lens.append(len(x))
            return torch.cat(chunks),torch.tensor(lens,device='cuda')
        fit=np.array([mapping[p] for p in splits['fit'] if cancers[mapping[p]]==cancer]);val=np.array([mapping[p] for p in splits['validation'] if cancers[mapping[p]]==cancer])
        outer=np.array([mapping[p] for p in splits['outer_train'] if cancers[mapping[p]]==cancer]);test=np.array([mapping[p] for p in splits['test'] if cancers[mapping[p]]==cancer])
        assert set(fit).isdisjoint(val) and set(outer).isdisjoint(test) and set(fit)|set(val)==set(outer)
        if a.smoke:fit=fit[:32];val=val[:16]
        def setup(rr):
            seed=2025+a.fold*101+idx['cancer_names'].index(cancer);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            mu=mom['mean'][rr].mean(0);sd=np.sqrt(np.maximum(mom['second'][rr].mean(0)-mu**2,0));sd=np.where(sd>1e-6,sd,1)
            model=classes[a.method](1536,len(genes),mu,sd,gene_y[rr].mean(0)).cuda()
            optimizer=(torch.optim.AdamW if a.method=='rnapath' else torch.optim.Adam)(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.1,patience=10) if a.method=='rnapath' else None
            return model,optimizer,scheduler
        def epoch(model,optimizer,rr,e):
            model.train();rng=np.random.default_rng(2025+a.fold*10000+e);order=rng.permutation(rr);losses=[]
            for s in range(0,len(order),cfg['batch']):
                ii=order[s:s+cfg['batch']];x,l=batch(ii,rng);yy=torch.as_tensor(gene_y[ii].copy(),device='cuda',dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,l)
                loss=(pred.float()-yy).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            return float(np.mean(losses))
        @torch.inference_mode()
        def predict(model,rr):
            model.eval();pp=[]
            for s in range(0,len(rr),cfg['batch']):
                x,l=batch(rr[s:s+cfg['batch']]);pp.append(model(x,l).float().cpu().numpy())
            return np.concatenate(pp)
        model,opt,sched=setup(fit);best=-np.inf;bestepoch=0;lr_schedule=[]
        for e in range(1,(1 if a.smoke else cfg['epochs'])+1):
            t=time.time();lr=opt.param_groups[0]['lr'];lr_schedule.append(lr);loss=epoch(model,opt,fit,e);vp=predict(model,val)
            vmse=float(np.mean((vp.astype(np.float64)-gene_y[val])**2));vr=genecorr(gene_y[val],vp)
            score=-vmse if a.method=='rnapath' else vr
            if score>best+1e-12:best=score;bestepoch=e
            row={'phase':'select','method':a.method,'fold':a.fold,'cancer':cancer,'epoch':e,'train_mse':loss,'val_mse':vmse,'val_gene_pearson':vr,'best_epoch':bestepoch,'lr':lr,'seconds':time.time()-t}
            with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if sched:sched.step(vmse)
            if e>=cfg['min_epochs'] and e-bestepoch>=cfg['patience']:break
        write(dest/'SELECTION.json',{'best_epoch':bestepoch,'best_score':best,'criterion':cfg['selection'],'lr_schedule':lr_schedule,'fit_n':len(fit),'validation_n':len(val),'outer_n':len(outer),'test_n':len(test),'protocol_sha256':sha(a.root/'PROTOCOL.json')})
        if a.smoke:return
        del model,opt,sched;torch.cuda.empty_cache();model,opt,sched=setup(outer)
        for e in range(1,bestepoch+1):
            for group in opt.param_groups:group['lr']=lr_schedule[e-1]
            t=time.time();loss=epoch(model,opt,outer,e)
            row={'phase':'refit','method':a.method,'fold':a.fold,'cancer':cancer,'epoch':e,'epochs':bestepoch,'loss':loss,'seconds':time.time()-t}
            with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        pp=predict(model,test)
        np.savez(dest/'TCGA_GENE.npz',patient_ids=np.array(idx['patient_ids'])[test],cancers=cancers[test],gene_ids=np.array(genes),prediction=pp)
        torch.save({'state_dict':model.state_dict(),'config':cfg,'gene_ids':genes,'selected_epoch':bestepoch},dest/'model.pt')
        er=np.where(np.array(ext['cancers'])==cancer)[0]
        if len(er):
            ep=[];model.eval()
            with torch.inference_mode():
                for start in range(0,len(er),cfg['batch']):
                    rr=er[start:start+cfg['batch']];ll=[int(extoff[i+1]-extoff[i]) for i in rr]
                    xx=torch.cat([torch.from_numpy(extfeat[extoff[i]:extoff[i+1]].copy()).cuda() for i in rr])
                    ep.append(model(xx,torch.tensor(ll,device='cuda')).float().cpu().numpy())
            np.savez(dest/'CPTAC_GENE.npz',patient_ids=np.array(ext['patient_ids'])[er],cancers=np.array(ext['cancers'])[er],gene_ids=np.array(genes),prediction=np.concatenate(ep))
        names=['TCGA_GENE.npz','model.pt','training.jsonl','SELECTION.json']+(['CPTAC_GENE.npz'] if len(er) else [])
        write(dest/'COMPLETE.json',{'status':'COMPLETE','method':a.method,'fold':a.fold,'cancer':cancer,'outputs':[ref(dest/n) for n in names]})
        del model,opt,sched,bank;gc.collect();torch.cuda.empty_cache()
    if len(selected)==len(idx['cancer_names']):
        assert all((out/c/'COMPLETE.json').exists() for c in idx['cancer_names'])
        write(out/'COMPLETE.json',{'status':'COMPLETE','method':a.method,'fold':a.fold,'cancers':idx['cancer_names'],'outputs':[ref(out/c/'COMPLETE.json') for c in idx['cancer_names']]})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--method',choices=['rnapath','path2omics'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--cancer');ap.add_argument('--smoke',action='store_true');main(ap.parse_args())
