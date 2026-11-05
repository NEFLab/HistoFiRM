"""Label-free complete-bag raw feature moments."""

import numpy as np
from .common import npz, digest, sha


def raw_cache(datasets, destination, chunk_size=1024):
    ids, means, seconds, signatures = [], [], [], []
    seen = {}
    for data in datasets:
        for i, pid in enumerate(data.ids):
            slides = data.rows[i]["slides"]
            identity = [(s["slide_id"], s["features"], s["n_patches"]) for s in slides]
            if pid in seen:
                if seen[pid] != identity:
                    raise ValueError("Same patient ID refers to different feature bags.")
                continue
            seen[pid] = identity
            n = 0
            total = np.zeros(data.manifest["input_dim"], dtype=np.float64)
            total2 = np.zeros_like(total)
            for chunk in data.chunks(i, chunk_size):
                x = chunk["x"].astype(np.float64)
                total += x.sum(0)
                total2 += (x * x).sum(0)
                n += len(x)
            if n == 0:
                raise ValueError("Empty feature bag")
            ids.append(pid)
            means.append(total / n)
            seconds.append(total2 / n)
            signatures.append(bag_signature(data, i))
    npz(
        destination,
        patient_ids=np.asarray(ids),
        mean=np.asarray(means),
        second=np.asarray(seconds),
        bag_signatures=np.asarray(signatures),
    )
    return load_cache(destination)


def load_cache(filename):
    with np.load(filename, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def bag_signature(data, index):
    return digest(
        [
            {
                "slide_id": s["slide_id"],
                "n_patches": s["n_patches"],
                "sha256": s["features"].get("sha256") or sha(s["features"]["path"]),
                "key": s["features"].get("key"),
                "row_start": s["features"].get("row_start"),
                "row_stop": s["features"].get("row_stop"),
            }
            for s in data.rows[index]["slides"]
        ]
    )


def validate_cache(data, cache):
    if "bag_signatures" not in cache or len(set(cache["patient_ids"])) != len(cache["patient_ids"]):
        raise ValueError("Feature cache lacks unique, verifiable bag identities.")
    index = {s: i for i, s in enumerate(cache["patient_ids"])}
    for i, pid in enumerate(data.ids):
        if pid not in index or bag_signature(data, i) != cache["bag_signatures"][index[pid]]:
            raise ValueError("Feature cache does not match the manifest.")
