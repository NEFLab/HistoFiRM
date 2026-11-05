"""Use exact classes from supplied author source; explicitly adapt AE decoder only.
No utilities, data loading, or source top-level GPU/device side effects are executed.
"""
import ast
from pathlib import Path
import torch
from torch import nn
from contracts import require,sha,load
ROOT=Path(__file__).resolve().parents[1]

def author_class(filename,name):
 p=ROOT/'vendor/deeppt'/filename
 pin=load(ROOT/'config/AUTHOR_SOURCE.json')['files'][filename]
 require(sha(p)==pin['sha256'],'Author source differs')
 tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name];require(len(nodes)==1,'Author class not uniquely located')
 env={'torch':torch,'nn':nn};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 return env[name]

def build_ae(input_dim=1536,hidden=512):
 a=author_class('model_AE.py','AutoEncoder')(input_dim,hidden,input_dim)
 require(isinstance(a.decoder[-1],nn.ReLU),'Unexpected upstream AE decoder; stop')
 a.decoder[-1]=nn.Identity() # Formal UNI2H-signed-feature adaptation; encoder unchanged.
 return a

def build_reg(outputs=18977,hidden=512,dropout=.2):
 cls=author_class('model_MLP.py','MLP_regression');m=cls(hidden,hidden,outputs,dropout,nn.Parameter(torch.zeros(outputs)))
 require(not any(isinstance(x,nn.ReLU) for x in m.modules()),'Unexpected nonlinear author regression; efficient pooling invalid')
 return m

def pooled_reg(model,encoded_chunks):
 """Mean hidden activations BEFORE final affine map. Keeps original per-tile dropout.
 It does not replace tile dropout by dropout of the mean feature. In eval this is
 algebraically equal to author layer1(layer0(all_tiles)).mean(0), up to rounding.
 """
 total=None;n=0
 for x in encoded_chunks:
  require(x.ndim==2 and len(x)>0,'Invalid encoded bag');h=model.layer0(x);s=h.sum(0);total=s if total is None else total+s;n+=len(x)
 require(n>0,'Empty patient bag');return model.layer1(total/n)
