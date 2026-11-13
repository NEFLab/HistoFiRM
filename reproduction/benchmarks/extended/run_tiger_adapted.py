"""Independent TIGER/HistoBistro transformer adaptation for fixed bulk targets.

The architecture is checked against the cited author's forward implementation.
UNI2-h inputs and target/split harmonisation are specified in PROTOCOL.json.
"""
from pathlib import Path
import argparse,json,hashlib,random,time,sys,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_mosby_adapted import load,write,sha,ref

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(512);self.qkv=nn.Linear(512,1536,bias=False);self.out=nn.Linear(512,512)
        self.n2=nn.LayerNorm(512);self.ff=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512))
    def forward(self,x,mask):
        q,k,v=self.qkv(self.n1(x)).chunk(3,-1)
        q,k,v=[a.reshape(a.shape[0],a.shape[1],8,64).transpose(1,2) for a in [q,k,v]]
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
        x=x+self.out(a.transpose(1,2).reshape(x.shape));return x+self.ff(self.n2(x))
class Net(nn.Module):
    def __init__(self,mu,sd):
        super().__init__();self.register_buffer('mu',torch.tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.tensor(sd,dtype=torch.float32))
        self.projection=nn.Sequential(nn.Linear(1536,512),nn.ReLU());self.cls=nn.Parameter(torch.randn(1,1,512))
        self.blocks=nn.ModuleList([Block(),Block()]);self.norm=nn.LayerNorm(512);self.head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1564))
    def forward(self,x,mask):
        x=self.projection((x.float()-self.mu)/self.sd);x=torch.cat((self.cls.expand(x.shape[0],-1,-1),x),1)
        mask=F.pad(mask,(1,0),value=True)
        for block in self.blocks:x=block(x,mask)
        return self.head(self.norm(x[:,0]))

def prepare(cache,out):
    assert (cache/'PREPARED.json').exists();out.mkdir(exist_ok=False)
    write(out/'PROTOCOL.json',{'method':'TIGER (UNI2-h, direct functional adaptation)',
      'source_repository':'https://github.com/fmhoward/TIGER','source_commit':'bdee92e6cfa96e9ddb5e3b48ab4fff0779a7bb4c',
      'paper_doi':'10.1101/2025.08.25.25334393','publication_status':'preprint per author repository at access date',
      'implementation':'independent implementation of documented two-block HistoBistro transformer; not authors full pipeline',
      'architecture':{'projection':[1536,512,'ReLU'],'depth':2,'heads':8,'head_dim':64,'feedforward':512,'pool':'CLS','position_encoding':None,'dropout':0},
      'source_config':'UNI2 TIGER/mil_params.json: 9 epochs, lr 0.001, wd 1e-5, batch 64, bag size 512, one-cycle',
      'max_epochs':9,'lr':0.001,'wd':0.00001,'batch':64,'microbatch':8,'training_tiles':512,
      'adaptations':['fixed 1564 signed functional targets replace original 775 signatures',
        'pooled TCGA16 training, cancer-balanced masked MSE and training-only within-cancer target standardization',
        'training-only patient-balanced feature standardization',
        'same archived five-fold fit/validation/test assignments; validation weighted MSE selects epochs',
        'random training subsample up to 512 tiles; shorter bags padded and attention-masked',
        'evaluation uses all retained patches, with inherited fixed 8000 per-slide cap',
        'batch 64 implemented by microbatch accumulation; torch SDPA preserves attention equation',
        'outer training refit uses selected epoch count and original 9-epoch one-cycle schedule prefix',
        'CPTAC average of five decoded predictions; no external labels in training'],
      'cache':str(cache),'cache_receipt_sha256':sha(cache/'PREPARED.json'),'seed':2025,
      'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    write(out/'PREPARED.json',{'status':'COMPLETE','protocol':ref(out/'PROTOCOL.json'),'code':ref(Path(__file__))})

def train(base,cache,out,fold,smoke=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    cfg=load(out/'PROTOCOL.json');sp=load(cache/'SPLITS.json')[str(fold)];idx=load(cache/'TCGA16_index.json')
    names=idx['cancer_names'];by={s:i for i,s in enumerate(idx['patient_ids'])};cancers=np.array(idx['cancers']);cc=np.array([names.index(s) for s in cancers]);off=np.array(idx['offsets'])
    raw=np.load(cache/'TCGA16_features.npy',mmap_mode='r');mom=np.load(cache/'TCGA16_moments.npz')
    dest=out/('smoke'+str(fold) if smoke else f'F{fold}');dest.mkdir(exist_ok=False)
    bank=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
    for i in range(0,len(raw),32000):bank[i:i+32000]=torch.from_numpy(raw[i:i+32000].copy()).cuda()
    def setup(part):
        r=load(base/f'inputs/folds/F{fold}/{part}.json');assert sha(r['labels']['path'])==r['labels']['sha256']
        a=np.load(r['labels']['path']);assert a['patient_ids'].tolist()==sp[part] and a['target_ids'].tolist()==idx['target_ids']
        ids=np.array([by[x] for x in sp[part]]);y=a['truth'];ci=cc[ids]
        center=np.zeros((16,1564));scale=np.ones_like(center);valid=np.zeros_like(center,dtype=bool);support=valid.copy()
        for j in range(16):
            z=y[ci==j];fin=np.isfinite(z);n=fin.sum(0);mu=np.divide(np.where(fin,z,0).sum(0),n,out=np.zeros(1564),where=n>0)
            ss=np.where(fin,z-mu,0);var=(ss*ss).sum(0);sd=np.sqrt(np.divide(var,n-1,out=np.zeros(1564),where=n>1))
            center[j]=mu;scale[j]=np.where(sd>1e-10,sd,1);valid[j]=(n>=2)&(var>1e-20);support[j]=n>=2
        mask=np.isfinite(y)&valid[ci];z=np.where(mask,(y-center[ci])/scale[ci],0).astype(np.float32)
        weights=len(ids)/(16*np.bincount(ci,minlength=16)[ci]);mu=mom['mean'][ids].mean(0);sd=np.sqrt(np.maximum(mom['second'][ids].mean(0)-mu*mu,0));sd=np.where(sd>1e-6,sd,1)
        torch.manual_seed(2025+fold);torch.cuda.manual_seed_all(2025+fold);np.random.seed(2025+fold);random.seed(2025+fold)
        net=Net(mu,sd).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],epochs=cfg['max_epochs'],steps_per_epoch=math.ceil(len(ids)/cfg['batch']),pct_start=.25,div_factor=25,final_div_factor=1e4)
        return {'ids':ids,'z':z,'mask':mask,'weights':weights,'net':net,'opt':opt,'sched':sched,'center':center,'scale':scale,'valid':valid,'support':support}
    def batch(rows,source,offset,rng=None):
        pieces=[]
        for i in rows:
            start,end=offset[i:i+2];a=source[start:end]
            if rng is not None and len(a)>cfg['training_tiles']:
                ix=rng.choice(len(a),cfg['training_tiles'],replace=False);a=a[torch.tensor(ix,device='cuda')]
            pieces.append(a)
        n=max(map(len,pieces));x=torch.zeros((len(rows),n,1536),dtype=torch.float16,device='cuda');mask=torch.zeros((len(rows),n),dtype=torch.bool,device='cuda')
        for i,a in enumerate(pieces):x[i,:len(a)]=a;mask[i,:len(a)]=True
        return x,mask
    @torch.inference_mode()
    def predict(net,rows,source,offset):
        net.eval();outpred=np.empty((len(rows),1564),np.float32)
        # Length buckets reduce padding; every retained tile is still evaluated.
        order=np.argsort([offset[i+1]-offset[i] for i in rows]);k=0
        while k<len(order):
            n=int(offset[rows[order[k]]+1]-offset[rows[order[k]]]);bs=8 if n<=1024 else (2 if n<=2048 else 1)
            pos=order[k:k+bs];x,m=batch(rows[pos],source,offset)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x,m)
            outpred[pos]=p.float().cpu().numpy();k+=len(pos)
        return outpred
    def epoch(pack,ep):
        net=pack['net'];net.train();rng=np.random.default_rng(2025+10000*fold+ep);order=rng.permutation(len(pack['ids']));losses=[]
        for start in range(0,len(order),cfg['batch']):
            pos=order[start:start+cfg['batch']];pack['opt'].zero_grad(set_to_none=True);total=0.
            for st in range(0,len(pos),cfg['microbatch']):
                ix=pos[st:st+cfg['microbatch']];x,m=batch(pack['ids'][ix],bank,off,rng)
                yy=torch.tensor(pack['z'][ix],device='cuda');mk=torch.tensor(pack['mask'][ix],device='cuda');ww=torch.tensor(pack['weights'][ix],dtype=torch.float32,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x,m)
                loss=((((p.float()-yy)**2)*mk).sum(1)/mk.sum(1).clamp_min(1)*ww).sum()/len(pos)
                if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
                loss.backward();total+=float(loss.detach())
            pack['opt'].step();pack['sched'].step();losses.append(total)
        return float(np.mean(losses))
    pack=setup('fit');vr=load(base/f'inputs/folds/F{fold}/validation.json');val=np.load(vr['labels']['path'])
    vrows=np.array([by[x] for x in val['patient_ids']]);vc=cc[vrows];vm=np.isfinite(val['truth'])&pack['valid'][vc]
    vz=np.where(vm,(val['truth']-pack['center'][vc])/pack['scale'][vc],0);vw=len(vrows)/(16*np.bincount(vc,minlength=16)[vc])
    best=np.inf;best_ep=0
    if smoke:
        for key in ['ids','z','mask','weights']:pack[key]=pack[key][:64]
        vrows=vrows[:32];vc=vc[:32];vm=vm[:32];vz=vz[:32];vw=vw[:32]
    for ep in range(1,(1 if smoke else cfg['max_epochs'])+1):
        t=time.time();loss=epoch(pack,ep);p=predict(pack['net'],vrows,bank,off);v=float(np.mean(((p-vz)**2*vm).sum(1)/np.maximum(vm.sum(1),1)*vw))
        if not np.isfinite(v):raise RuntimeError('INVALID_VALIDATION')
        if v<best:best=v;best_ep=ep
        row={'phase':'select','fold':fold,'epoch':ep,'loss':loss,'validation_mse':v,'best_epoch':best_ep,'seconds':time.time()-t}
        with open(dest/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write(dest/'SELECTION.json',{'epochs':best_ep,'validation_mse':best,'protocol_sha256':sha(out/'PROTOCOL.json')})
    if smoke:return
    del pack;torch.cuda.empty_cache();pack=setup('outer_train')
    for ep in range(1,best_ep+1):
        t=time.time();loss=epoch(pack,ep);row={'phase':'refit','fold':fold,'epoch':ep,'epochs':best_ep,'loss':loss,'seconds':time.time()-t}
        with open(dest/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    net=pack['net'];torch.save({'state_dict':net.state_dict(),'target_center':pack['center'],'target_scale':pack['scale'],'protocol':cfg,'epochs':best_ep},dest/'model.pt')
    def export(co,ids,cs,rows,source,offset):
        p=predict(net,rows,source,offset);ci=np.array([names.index(c) for c in cs]);y=p*pack['scale'][ci]+pack['center'][ci]
        y=np.where(pack['valid'][ci],y,pack['center'][ci]);y=np.where(pack['support'][ci],y,np.nan)
        np.savez(dest/f'{co}_prediction.npz',patient_ids=np.array(ids),cancers=np.array(cs),target_ids=np.array(idx['target_ids']),prediction=y)
    rows=np.array([by[x] for x in sp['test']]);export('TCGA16',sp['test'],cancers[rows].tolist(),rows,bank,off)
    del bank;torch.cuda.empty_cache();ci=load(cache/'CPTAC8_index.json');raw=np.load(cache/'CPTAC8_features.npy',mmap_mode='r');eb=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
    for i in range(0,len(raw),32000):eb[i:i+32000]=torch.from_numpy(raw[i:i+32000].copy()).cuda()
    export('CPTAC8',ci['patient_ids'],ci['cancers'],np.arange(len(ci['patient_ids'])),eb,np.array(ci['offsets']))
    write(dest/'COMPLETE.json',{'status':'COMPLETE','fold':fold,'selected_epochs':best_ep,'outputs':[ref(dest/n) for n in ['model.pt','TCGA16_prediction.npz','CPTAC8_prediction.npz','training.jsonl','SELECTION.json']]})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train','smoke']);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args()
    if a.mode=='prepare':prepare(a.cache,a.out)
    else:train(a.base,a.cache,a.out,a.fold,a.mode=='smoke')
