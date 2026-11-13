"""File and axis checks for the fixed functional scorer."""
from pathlib import Path
import hashlib
import json

class ContractError(ValueError):
    pass

def require(condition,message):
    if not condition:raise ContractError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_json(path):
    path=Path(path).resolve()
    def resolve(value):
        if isinstance(value,list):return [resolve(x) for x in value]
        if isinstance(value,dict):
            result={k:resolve(v) for k,v in value.items()}
            if isinstance(value.get('path'),str):
                p=Path(value['path'])
                result['path']=str(p if p.is_absolute() else (path.parent/p).resolve())
            return result
        return value
    return resolve(json.loads(path.read_text(encoding='utf-8')))

def verify_record(record):
    p=Path(record['path'])
    require(sha(p)==record['sha256'],'File checksum mismatch: '+p.name)
    return p

def unique(values,label='IDs'):
    require(len(values)==len(set(values)),'Duplicate '+label)
    return values

def same(a,b,label='axis'):
    require(list(a)==list(b),'Ordered '+label+' differs')
