"""Small, explicit IO utilities. No global server paths or cohort discovery."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np

class ContractError(RuntimeError): pass

def require(ok, message):
    if not ok: raise ContractError(message)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def json_safe(x):
    if isinstance(x,dict): return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x,np.ndarray): return json_safe(x.tolist())
    if isinstance(x,np.generic): return json_safe(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def write_json(path,x,replace=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    require(replace or not p.exists(),f'Output exists; reconcile instead of overwrite: {p}')
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(json_safe(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        if not replace: require(not p.exists(),f'Concurrent writer: {p}')
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def save_npz(path,**arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);require(not p.exists(),f'Output exists: {p}')
    for a in arrays.values():require(np.asarray(a).dtype.kind!='O','Object arrays are prohibited')
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**arrays)
        require(not p.exists(),'Concurrent writer');os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def record(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def verified(rec):
    p=Path(rec['path']);require(p.is_file(),f'Missing {p}')
    require(p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],f'File identity mismatch: {p}')
    return p

def unique(ids):require(len(ids)==len(set(map(str,ids))),'Duplicate IDs')
def same(a,b):require(np.array_equal(np.asarray(a).astype(str),np.asarray(b).astype(str)),'Ordered identities differ')
def disjoint(a,b):require(not set(map(str,a))&set(map(str,b)),'Train/test patient overlap')
