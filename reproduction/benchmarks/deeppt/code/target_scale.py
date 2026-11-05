from dataclasses import dataclass
import numpy as np
from contracts import require
@dataclass
class Scale:
 mean:np.ndarray
 sd:np.ndarray
 n:np.ndarray
 constant:np.ndarray
 @classmethod
 def fit(cls,y):
  y=np.asarray(y,float);m=np.isfinite(y);n=m.sum(0);a=np.divide(np.where(m,y,0).sum(0),n,out=np.zeros(y.shape[1]),where=n>0);v=np.divide(np.where(m,y-a,0).__pow__(2).sum(0),n,out=np.zeros(y.shape[1]),where=n>0);sd=np.sqrt(v);return cls(a,np.where(sd>1e-12,sd,1.),n,(n>=2)&(sd<=1e-12))
 @property
 def supported(self):return self.n>=2
 def transform(self,y):return np.where(self.supported & ~self.constant,(np.asarray(y)-self.mean)/self.sd,np.nan)
 def inverse(self,y):return np.where(self.supported,np.where(self.constant,self.mean,np.asarray(y)*self.sd+self.mean),np.nan)
 def data(self):return {'mean':self.mean,'sd':self.sd,'n':self.n,'constant':self.constant}
 @classmethod
 def from_data(cls,x):return cls(*(np.asarray(x[k]) for k in ['mean','sd','n','constant']))

def patient_masked_mse(pred,target):
 import torch
 mask=torch.isfinite(target);count=mask.sum(-1);require(bool((count>0).any()),'Batch has no supported labels')
 e=torch.where(mask,pred-torch.nan_to_num(target),0.).square().sum(-1)/count.clamp_min(1)
 return e[count>0].mean()
