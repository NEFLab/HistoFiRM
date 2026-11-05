"""Explicit patient and slide manifests; all paths resolve relative to the manifest."""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .common import load as load_json, sha as digest_file


def unique(values, label):
    if len(set(values)) != len(values):
        raise ValueError("Duplicate " + label)


class ArrayReader:
    def __init__(self, max_cache_bytes=512 << 20):
        self.cache = OrderedDict()
        self.bytes = 0
        self.cap = max_cache_bytes

    def _bounds(self, ref, shape):
        start = int(ref.get("row_start", 0))
        stop = int(ref.get("row_stop", shape[0]))
        if start < 0 or stop < start or stop > shape[0]:
            raise ValueError("ARRAY_ROW_BOUNDS")
        return (start, stop)

    def shape(self, ref):
        path = Path(ref["path"])
        key = ref.get("key")
        if path.suffix == ".npy":
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        elif path.suffix == ".npz":
            shape = self.read(ref).shape
            return shape
        else:
            import h5py

            if not key:
                raise ValueError("H5_KEY_REQUIRED")
            with h5py.File(path, "r") as f:
                shape = f[key].shape
        start, stop = self._bounds(ref, shape)
        return (stop - start, *shape[1:])

    def read(self, ref):
        path = str(ref["path"])
        key = ref.get("key")
        start = ref.get("row_start")
        stop = ref.get("row_stop")
        sl = slice(start, stop)
        cachekey = (path, key, start, stop)
        if cachekey in self.cache:
            a = self.cache.pop(cachekey)
            self.cache[cachekey] = a
            return a
        if Path(path).suffix == ".npy":
            base = np.load(path, mmap_mode="r", allow_pickle=False)
            self._bounds(ref, base.shape)
            a = base[sl]
        elif Path(path).suffix == ".npz":
            if not key:
                raise ValueError("NPZ_KEY_REQUIRED")
            with np.load(path, allow_pickle=False) as f:
                base = f[key]
                self._bounds(ref, base.shape)
                a = np.asarray(base[sl])
        else:
            import h5py

            if not key:
                raise ValueError("H5_KEY_REQUIRED")
            with h5py.File(path, "r") as f:
                self._bounds(ref, f[key].shape)
                a = np.asarray(f[key][sl])
        if a.dtype.kind == "O":
            raise ValueError("OBJECT_ARRAY_NOT_ALLOWED")
        if a.nbytes <= self.cap:
            while self.cache and self.bytes + a.nbytes > self.cap:
                _, old = self.cache.popitem(last=False)
                self.bytes -= old.nbytes
            self.cache[cachekey] = a
            self.bytes += a.nbytes
        return a

    def read_slice(self, ref, start, stop):
        path = Path(ref["path"])
        key = ref.get("key")
        start = int(start)
        stop = int(stop)
        if path.suffix in (".npy", ".npz"):
            a = self.read(ref)
            if start < 0 or stop < start or stop > len(a):
                raise ValueError("SLICE_BOUNDS")
            return a[start:stop]
        import h5py

        if not key:
            raise ValueError("H5_KEY_REQUIRED")
        with h5py.File(path, "r") as f:
            ds = f[key]
            base, end = self._bounds(ref, ds.shape)
            if start < 0 or stop < start or stop > end - base:
                raise ValueError("SLICE_BOUNDS")
            a = np.asarray(ds[base + start : base + stop])
        if a.dtype.kind == "O":
            raise ValueError("OBJECT_ARRAY_NOT_ALLOWED")
        return a

    def read_rows(self, ref, indices):
        ix = np.asarray(indices)
        if ix.ndim != 1 or ix.dtype.kind not in "iu":
            raise ValueError("ROW_INDEX_TYPE")
        path = Path(ref["path"])
        key = ref.get("key")
        if path.suffix in (".npy", ".npz"):
            a = self.read(ref)
            if len(ix) and (ix.min() < 0 or ix.max() >= len(a)):
                raise ValueError("ROW_INDEX_BOUNDS")
            return np.asarray(a[ix])
        import h5py

        if not key:
            raise ValueError("H5_KEY_REQUIRED")
        with h5py.File(path, "r") as f:
            ds = f[key]
            base, end = self._bounds(ref, ds.shape)
            if len(ix) and (ix.min() < 0 or ix.max() >= end - base):
                raise ValueError("ROW_INDEX_BOUNDS")
            unique_ix, inverse = np.unique(ix, return_inverse=True)
            a = np.asarray(ds[base + unique_ix.astype(np.int64)])[inverse]
        if a.dtype.kind == "O":
            raise ValueError("OBJECT_ARRAY_NOT_ALLOWED")
        return a


class Patients:
    def __init__(self, path, labels_required=False, reader=None):
        self.path = Path(path)
        self.manifest = load_json(path)
        for patient in self.manifest["patients"]:
            for slide in patient["slides"]:
                for key in ("features", "coords"):
                    if key in slide:
                        r = slide[key]
                        q = Path(r["path"])
                        r["path"] = str(q if q.is_absolute() else self.path.parent / q)
                        if "sha256" in r and digest_file(r["path"]) != r["sha256"]:
                            raise ValueError("Array hash mismatch")
        if "labels" in self.manifest:
            r = self.manifest["labels"]
            q = Path(r["path"])
            r["path"] = str(q if q.is_absolute() else self.path.parent / q)
        self.rows = self.manifest["patients"]
        self.cohort = self.manifest["cohort"]
        self.reader = reader or ArrayReader()
        self.ids = [p["patient_id"] for p in self.rows]
        unique(self.ids, "PATIENT")
        self.cancers = [p["cancer"] for p in self.rows]
        self.target_ids = list(self.manifest["target_ids"])
        unique(self.target_ids, "TARGET")
        self.cancer_names = list(self.manifest["cancer_names"])
        unique(self.cancer_names, "CANCER")
        self.source_ids = list(self.manifest["source_ids"])
        if len(self.source_ids) != len(self.target_ids):
            raise ValueError("SOURCE_ORDER")
        self.labels = None
        if "labels" in self.manifest:
            r = self.manifest["labels"]
            if digest_file(r["path"]) != r["sha256"]:
                raise ValueError("LABEL_HASH")
            with np.load(r["path"], allow_pickle=False) as f:
                if (
                    f["patient_ids"].tolist() != self.ids
                    or f["target_ids"].tolist() != self.target_ids
                ):
                    raise ValueError("LABEL_AXIS")
                if "cancers" not in f or f["cancers"].tolist() != self.cancers:
                    raise ValueError("LABEL_CANCER_AXIS")
                self.labels = np.asarray(f["truth"], dtype=np.float64)
        if labels_required and self.labels is None:
            raise ValueError("LABELS_REQUIRED")
        if self.labels is not None and self.labels.shape != (len(self.rows), len(self.target_ids)):
            raise ValueError("LABEL_SHAPE")
        for p in self.rows:
            if p["cancer"] not in self.cancer_names:
                raise ValueError("UNKNOWN_CANCER")
            if not p["slides"]:
                raise ValueError("EMPTY_PATIENT")
            unique([s["slide_id"] for s in p["slides"]], "SLIDE")
            if any((int(s["n_patches"]) < 1 for s in p["slides"])):
                raise ValueError("EMPTY_SLIDE")

    def n_patches(self, index):
        return sum((int(s["n_patches"]) for s in self.rows[index]["slides"]))

    def chunks(self, index, chunk_size=1024, with_coords=False):
        if chunk_size < 1:
            raise ValueError("CHUNK_SIZE")
        offset = 0
        for s in self.rows[index]["slides"]:
            n = int(s["n_patches"])
            if self.reader.shape(s["features"]) != (n, self.manifest["input_dim"]):
                raise ValueError("FEATURE_SHAPE")
            if with_coords and self.reader.shape(s["coords"]) != (n, 2):
                raise ValueError("COORD_AXIS")
            for start in range(0, n, chunk_size):
                stop = min(n, start + chunk_size)
                x = np.asarray(self.reader.read_slice(s["features"], start, stop), dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError("NONFINITE_FEATURE")
                coords = self.reader.read_slice(s["coords"], start, stop) if with_coords else None
                yield {
                    "x": x,
                    "mass": np.ones(stop - start, dtype=np.float64),
                    "indices": np.arange(offset + start, offset + stop),
                    "slide_id": s["slide_id"],
                    "coords": None if coords is None else np.asarray(coords, dtype=np.float64),
                }
            offset += n
