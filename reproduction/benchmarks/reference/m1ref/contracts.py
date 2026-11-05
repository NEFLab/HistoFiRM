"""Narrow data contracts and atomic evidence. These checks do not replace OS isolation."""
from __future__ import annotations
import contextlib, hashlib, json, os, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np
from . import PROTOCOL_ID

class ContractError(RuntimeError): pass
class BudgetHold(RuntimeError): pass

def require(ok: bool, message: str) -> None:
    if not ok: raise ContractError(message)

def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def clean(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,ensure_ascii=False,
                                     separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding='utf-8') as f: return json.load(f)

def atomic_bytes(path: str | Path, payload: bytes) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.exists(), f'Immutable output already exists: {p}')
    fd,tmp=tempfile.mkstemp(prefix='.partial_',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(payload);f.flush();os.fsync(f.fileno())
        os.link(tmp,p)  # fails if another writer created p; no silent overwrite
    finally:
        if Path(tmp).exists(): Path(tmp).unlink()

def write_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path,(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())

def save_npz(path: str | Path, **arrays: Any) -> None:
    import io
    buf=io.BytesIO();np.savez(buf,**arrays);atomic_bytes(path,buf.getvalue())

def record(path: str | Path) -> dict:
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def verify_record(ref: dict) -> Path:
    require(set(('path','sha256')).issubset(ref), 'Asset needs actual path and SHA')
    p=Path(ref['path']);require(p.is_file(),f'Asset missing: {p}')
    require(sha(p)==ref['sha256'],f'Asset changed: {p}')
    if 'bytes' in ref: require(p.stat().st_size==ref['bytes'],'Asset size mismatch')
    return p

def unique(values, name='IDs') -> np.ndarray:
    a=np.asarray(values).astype(str)
    require(a.ndim==1 and len(a)>0 and np.all(np.char.str_len(a)>0),f'Empty/invalid {name}')
    require(len(set(a.tolist()))==len(a),f'Duplicate {name}')
    return a

def same(a,b,name='ordered identity'):
    require(np.array_equal(np.asarray(a).astype(str),np.asarray(b).astype(str)),f'Mismatch: {name}')

def disjoint(*groups):
    seen=set()
    for group in groups:
        ids=set(map(str,group));require(not seen.intersection(ids),'Patient leakage/overlapping roles');seen.update(ids)

def load_prediction(path: str | Path) -> dict:
    with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
    allowed={'patient_id','cancer','target_id','prediction'}
    require(set(d)==allowed,'Prediction artifact must contain only IDs, cancer, targets, prediction')
    unique(d['patient_id']);unique(d['target_id'],'targets')
    require(d['prediction'].shape==(len(d['patient_id']),len(d['target_id'])),'Prediction shape mismatch')
    require(np.isfinite(d['prediction']).all(),'Nonfinite predictions; do not discard rows')
    return d

def check_release(path: str | Path, action: str) -> dict:
    r=read_json(path)
    require(r.get('protocol_id')==PROTOCOL_ID,'Wrong protocol')
    require(r.get('release')=='M1A_ONLY','Only M1A is released by this package')
    require(action in r['allowed_actions'],f'Action not authorized: {action}')
    return r

@contextlib.contextmanager
def exclusive_lock(path):
    import fcntl
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise ContractError('Another writer owns this logical unit') from e
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

class Clock:
    def __init__(self, seconds: float):
        require(seconds>0,'Time budget must be positive');self.start=time.monotonic();self.seconds=seconds
    def check(self):
        if time.monotonic()-self.start>=self.seconds: raise BudgetHold('Unit budget reached; partial is not COMPLETE')
    def elapsed(self): return time.monotonic()-self.start
