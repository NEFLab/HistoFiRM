"""I/O and lineage contracts; OS isolation is supplied by the existing server launcher."""
from pathlib import Path
import hashlib,json,os,tempfile,math
import numpy as np

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def load(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def record(p):return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def verified(r):
 p=Path(r['path']);require(p.is_file(),'Missing bound input: '+str(p));require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Asset identity mismatch '+str(p));return p
def write(p,obj,replace=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);require(replace or not p.exists(),'Existing artifact; reconcile, do not overwrite '+str(p));fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 try:
  with os.fdopen(fd,'w') as f:json.dump(clean(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def disjoint(a,b):require(not (set(a)&set(b)),'Patient sets overlap')
def unique(a):require(len(set(a))==len(a),'Duplicate patient/axis')
def save_npz(p,**kwargs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);require(not p.exists(),'Unregistered existing prediction; reconcile');fd,t=tempfile.mkstemp(dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**kwargs)
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
