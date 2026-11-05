import hashlib, json, os, time, uuid
from pathlib import Path
import numpy as np


def load(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ref(p):
    p = Path(p).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def verify(r):
    p = Path(r["path"])
    if not p.is_file() or p.stat().st_size != r["bytes"] or sha(p) != r["sha256"]:
        raise ValueError("SOURCE_CHANGED: " + str(p))
    return p


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def digest(o):
    return hashlib.sha256(
        json.dumps(clean(o), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write(p, o, immutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    o = clean(o)
    if immutable and p.exists():
        if load(p) != o:
            raise ValueError("IMMUTABLE_CONFLICT: " + str(p))
        return p
    q = p.with_name(p.name + ".tmp." + uuid.uuid4().hex)
    q.write_text(
        json.dumps(o, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8"
    )
    os.replace(q, p)
    return p


def npz(p, **arrays):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp." + uuid.uuid4().hex)
    with q.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(q, p)
    return p


def save_torch(p, o):
    import torch

    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp." + uuid.uuid4().hex)
    torch.save(o, q)
    os.replace(q, p)


def seed_int(*parts):
    return int(digest(parts)[:15], 16) % (2**32 - 1)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def schema_axes(d):
    return {"patient_ids": d.ids, "cancers": d.cancers, "target_ids": d.target_ids}


def load_prediction(p):
    with np.load(p, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files}
    for k in ["patient_ids", "cancers", "target_ids"]:
        if k not in out:
            raise ValueError("PREDICTION_AXIS_MISSING " + k)
    if "prediction" not in out:
        raise ValueError("PREDICTION_VALUES_MISSING")
    if len(set(out["patient_ids"].tolist())) != len(out["patient_ids"]):
        raise ValueError("DUPLICATE_PATIENT")
    return out


def require(condition, message):
    if not condition:
        raise ValueError(message)
