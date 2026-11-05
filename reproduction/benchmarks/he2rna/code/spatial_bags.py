"""B3 spatial supertile bags. No feature-space tokens or target access."""
from collections import OrderedDict
import numpy as np
import torch
from common import require,verified,same

class SpatialBags:
    def __init__(self, patients, device, cache_bytes=1536*1024**2):
        self.records=patients;self.device=device;self.cache=OrderedDict();self.cache_bytes=cache_bytes;self.used=0
        for p in patients:
            require(set(p)>={'patient_id','cancer','supertiles'},'Unbound patient supertiles')
            verified(p['supertiles'])
    def get_cpu(self,i):
        i=int(i)
        if i in self.cache:
            v=self.cache.pop(i);self.cache[i]=v;return v
        r=self.records[i]
        with np.load(r['supertiles']['path'],allow_pickle=False) as z:
            require(set(z.files)=={'supertiles','valid_lengths','patch_counts','slide_id','patient_id'},'Unexpected spatial cache fields')
            same(z['patient_id'],[r['patient_id']]);x=z['supertiles'].copy();n=z['valid_lengths'].astype(np.int64);w=z['patch_counts'].astype(np.int64)
            require(x.ndim==3 and x.shape[2]==1536 and 1<=x.shape[1]<=100,'Invalid spatial supertile shape')
            require(n.shape==w.shape==(x.shape[0],) and np.all((n>0)&(n<=x.shape[1])) and np.all(w>=n),'Invalid spatial lengths or patch counts')
            require(len(z['slide_id'])==len(x) and len(set(z['slide_id'].astype(str)))==len(x),'Slide identity invalid')
            require(np.isfinite(x).all(),'Nonfinite spatial features')
        v=(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(n,dtype=torch.long),torch.as_tensor(w,dtype=torch.float32))
        size=sum(t.numel()*t.element_size() for t in v)
        while self.cache and self.used+size>self.cache_bytes:
            _,old=self.cache.popitem(last=False);self.used-=sum(t.numel()*t.element_size() for t in old)
        if size<=self.cache_bytes:self.cache[i]=v;self.used+=size
        return v
    def batch(self,indices):
        bags=[self.get_cpu(i) for i in indices];mx=max(int(x.shape[1]) for x,_,_ in bags)
        xs=[];ns=[];ws=[];ps=[]
        for p,(x,n,w) in enumerate(bags):
            if x.shape[1]<mx:
                x=torch.nn.functional.pad(x,(0,0,0,mx-x.shape[1]))
            xs.append(x);ns.append(n);ws.append(w);ps.append(torch.full((len(x),),p,dtype=torch.long))
        return tuple(torch.cat(v).to(self.device) for v in (xs,ns,ps,ws))
    def forward(self,model,indices):
        x,n,p,w=self.batch(indices)
        return model.forward_patients(x,n,p,w,num_patients=len(indices))
