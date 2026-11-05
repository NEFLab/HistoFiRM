"""Matched reference regressors. None is labelled as an author model or frozen HCFR."""
from __future__ import annotations
import copy
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .contracts import require,Clock

class Scale:
    def fit(self,x,allow_missing=False):
        a=np.asarray(x,dtype=np.float64)
        require(a.ndim==2 and not np.isinf(a).any(),'Bad matrix')
        if not allow_missing: require(np.isfinite(a).all(),'Missing X is not allowed')
        require(np.all(np.isfinite(a).sum(0)>=2),'Fewer than two fit values in column')
        self.mean=np.nanmean(a,0);self.sd=np.nanstd(a,0,ddof=0)
        self.constant=self.sd<1e-12;self.sd[self.constant]=1.
        return self
    def transform(self,x):
        a=np.asarray(x,dtype=np.float64);require(a.shape[-1]==len(self.mean),'Scaler shape mismatch')
        return (a-self.mean)/self.sd
    def inverse(self,x): return np.asarray(x,dtype=np.float64)*self.sd+self.mean

def scaled_mse(pred,y,scale):
    good=np.isfinite(y);require(good.any(),'No evaluable validation values')
    require(np.isfinite(np.asarray(pred)[good]).all(),'Nonfinite validation prediction')
    row_n=good.sum(1);use=row_n>0
    sq=np.where(good,((np.asarray(pred)-y)/scale.sd)**2,0.)
    return float(np.mean(sq.sum(1)[use]/row_n[use]))

class Ridge:
    """Shared multi-output ridge with grouped missingness and bounded output solves."""
    def __init__(self,alpha=100.,output_chunk=512):
        require(alpha>0,'alpha>0 required');self.alpha=float(alpha);self.output_chunk=output_chunk
    def fit(self,x,y):
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        require(len(x)==len(y),'X/Y row mismatch')
        self.xscale=Scale().fit(x);self.yscale=Scale().fit(y,True)
        z=self.xscale.transform(x);v=self.yscale.transform(y)
        self.coef=np.zeros((x.shape[1],y.shape[1]),dtype=np.float64);self.intercept=np.zeros(y.shape[1])
        groups={}
        for j in range(y.shape[1]):
            mask=np.isfinite(y[:,j]);key=np.packbits(mask).tobytes()
            groups.setdefault(key,[mask,[]])[1].append(j)
        for mask,cols in groups.values():
            require(mask.sum()>=2,'Target has insufficient fitting labels')
            a=z[mask];xm=a.mean(0);a=a-xm
            dual=len(a)<a.shape[1]
            gram=a@a.T if dual else a.T@a
            gram.flat[::len(gram)+1]+=self.alpha
            factor=cho_factor(gram,lower=True,check_finite=False)
            for start in range(0,len(cols),self.output_chunk):
                js=np.array(cols[start:start+self.output_chunk]);b=v[np.ix_(mask,js)];ym=b.mean(0);b-=ym
                coef=a.T@cho_solve(factor,b,check_finite=False) if dual else cho_solve(factor,a.T@b,check_finite=False)
                self.coef[:,js]=coef;self.intercept[js]=ym-xm@coef
        self.output_dim=y.shape[1];return self
    def predict(self,x):
        z=self.xscale.transform(x);out=np.empty((len(z),self.output_dim),dtype=np.float64)
        for s in range(0,self.output_dim,self.output_chunk):
            e=min(s+self.output_chunk,self.output_dim)
            out[:,s:e]=(z@self.coef[:,s:e]+self.intercept[s:e])*self.yscale.sd[s:e]+self.yscale.mean[s:e]
        return out

class ScopeRidge:
    """New scope comparator, never called the frozen SAME_CHECKPOINT_ANCHOR."""
    def __init__(self,scope='shared',alpha=100.,cancer_alpha=500.,shrinkage=50.):
        require(scope in ['shared','per_cancer','hierarchical'],'Unknown scope')
        self.scope,self.alpha,self.cancer_alpha,self.shrinkage=scope,float(alpha),float(cancer_alpha),float(shrinkage)
    def fit(self,x,y,cancer):
        x=np.asarray(x);y=np.asarray(y);c=np.asarray(cancer)
        self.local={};self.global_model=None
        if self.scope!='per_cancer':self.global_model=Ridge(self.alpha).fit(x,y)
        target=y if self.scope=='per_cancer' else y-self.global_model.predict(x)
        if self.scope!='shared':
            for group in sorted(set(c)):
                mask=c==group;n=int(mask.sum())
                m=Ridge(self.alpha if self.scope=='per_cancer' else self.cancer_alpha).fit(x[mask],target[mask])
                w=1. if self.scope=='per_cancer' else n/(n+self.shrinkage)
                self.local[group]=(w,m)
        self.output_dim=y.shape[1];return self
    def predict(self,x,cancer):
        c=np.asarray(cancer);x=np.asarray(x)
        if self.scope=='per_cancer': require(set(c).issubset(self.local),'Unseen cancer in per-cancer model')
        out=np.zeros((len(x),self.output_dim)) if self.global_model is None else self.global_model.predict(x)
        for group,(w,m) in self.local.items():
            mask=c==group
            if mask.any():out[mask]+=w*m.predict(x[mask])
        return out

def make_mlp(input_dim,output_dim,hidden=(512,256),dropout=.1):
    import torch.nn as nn
    layers=[];d=input_dim
    for h in hidden:layers.extend([nn.Linear(d,h),nn.ReLU(),nn.Dropout(dropout)]);d=h
    layers.append(nn.Linear(d,output_dim))
    return nn.Sequential(*layers)

class MLP:
    def __init__(self,config,seed=2025,device='cpu'):
        self.cfg=dict(config);self.seed=int(seed);self.device=str(device)
    def _init(self,x,y):
        import torch
        require(self.device=='cpu' or torch.cuda.is_available(),'Requested CUDA device unavailable')
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark=False
        if hasattr(torch.backends,'cuda'):torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.xscale=Scale().fit(x);self.yscale=Scale().fit(y,True)
        self.input_dim=x.shape[1];self.output_dim=y.shape[1]
        self.net=make_mlp(self.input_dim,self.output_dim,self.cfg['hidden'],self.cfg['dropout']).to(self.device)
        return torch.optim.AdamW(self.net.parameters(),lr=self.cfg['lr'],weight_decay=self.cfg['weight_decay'])
    @staticmethod
    def loss(pred,target):
        import torch
        good=torch.isfinite(target);count=good.sum(1);valid=count>0
        require(bool(valid.any()),'Batch without labels')
        safe=torch.where(good,target,torch.zeros_like(target))
        loss=torch.where(good,(pred-safe)**2,torch.zeros_like(pred)).sum(1)/count.clamp_min(1)
        return loss[valid].mean()
    def train(self,x,y,clock,valid=None,epochs=None):
        """valid is ONLY inner TCGA. Final refit supplies fixed epochs and no valid data."""
        import torch
        opt=self._init(np.asarray(x),np.asarray(y));cfg=self.cfg
        xx=torch.as_tensor(self.xscale.transform(x),dtype=torch.float32)
        yy=torch.as_tensor(self.yscale.transform(y),dtype=torch.float32)
        gen=torch.Generator().manual_seed(self.seed+9281)
        n_epochs=cfg['max_epochs'] if valid is not None else int(epochs)
        require(n_epochs>0 and n_epochs<=cfg['max_epochs'],'Invalid frozen epoch budget')
        self.history=[];best_loss=np.inf;best_state=None;wait=0;self.best_epoch=0
        for ep in range(n_epochs):
            clock.check();self.net.train();loss_sum=0.;seen=0
            for ids in torch.randperm(len(xx),generator=gen).split(cfg['batch_size']):
                clock.check();bx=xx[ids].to(self.device);by=yy[ids].to(self.device)
                opt.zero_grad(set_to_none=True);loss=self.loss(self.net(bx),by)
                require(bool(torch.isfinite(loss)),'Nonfinite loss')
                loss.backward();opt.step();loss_sum+=float(loss.detach().cpu())*len(ids);seen+=len(ids)
            rec={'epoch_1based':ep+1,'train_scaled_mse':loss_sum/seen}
            if valid is not None:
                vl=scaled_mse(self.predict(valid[0]),np.asarray(valid[1]),self.yscale);rec['inner_validation_scaled_mse']=vl
                require(np.isfinite(vl),'Nonfinite validation loss')
                if vl<best_loss:
                    best_loss=vl;self.best_epoch=ep+1;best_state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()};wait=0
                else:wait+=1
            self.history.append(rec)
            if valid is not None and wait>=cfg['patience']:break
        if valid is not None:
            require(best_state is not None,'No selected epoch');self.net.load_state_dict(best_state)
        else:self.best_epoch=n_epochs
        self.net.eval();return self
    def predict(self,x):
        import torch
        self.net.eval();result=[]
        with torch.no_grad():
            for s in range(0,len(x),self.cfg['batch_size']):
                a=torch.as_tensor(self.xscale.transform(x[s:s+self.cfg['batch_size']]),dtype=torch.float32,device=self.device)
                result.append(self.net(a).cpu().numpy())
        return self.yscale.inverse(np.concatenate(result))
    def portable(self):
        self.net=self.net.cpu();self.device='cpu';return self
