"""Generate synthetic feature bags for a CPU installation check."""

import argparse
from pathlib import Path
import numpy as np
from histofirm.common import write, npz, sha


def make_demo(destination):
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose an empty demo directory.")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    targets = ["HALLMARK::DEMO_A", "HALLMARK::DEMO_B", "DEMO::C"]
    cancers = ["DEMO_1", "DEMO_2"]
    coef = rng.normal(size=(6, 3))
    manifests = {}
    for role, n in [("fit", 16), ("validation", 12), ("test", 10)]:
        rows = []
        ys = []
        for c in cancers:
            for i in range(n):
                pid = f"{role}_{c}_{i:02d}"
                x = rng.normal(size=(19 + i % 7, 6)).astype("float32")
                fn = root / (pid + ".npy")
                np.save(fn, x)
                rows.append(
                    {
                        "patient_id": pid,
                        "cancer": c,
                        "outer_fold": 0 if role == "test" else 1,
                        "slides": [
                            {
                                "slide_id": pid + "_slide",
                                "n_patches": len(x),
                                "features": {"path": fn.name, "sha256": sha(fn)},
                            }
                        ],
                    }
                )
                ys.append(x.mean(0) @ coef + rng.normal(scale=0.1, size=3))
        labels = root / (role + "_labels.npz")
        npz(
            labels,
            truth=np.asarray(ys),
            patient_ids=np.asarray([r["patient_id"] for r in rows]),
            target_ids=np.asarray(targets),
            cancers=np.asarray([r["cancer"] for r in rows]),
        )
        manifest = {
            "cohort": "SYNTHETIC",
            "input_dim": 6,
            "patients": rows,
            "target_ids": targets,
            "source_ids": ["HALLMARK", "HALLMARK", "DEMO"],
            "cancer_names": cancers,
            "labels": {"path": labels.name, "sha256": sha(labels)},
        }
        write(root / (role + ".json"), manifest)
        manifests[role] = manifest
    outer = {
        **manifests["fit"],
        "patients": manifests["fit"]["patients"] + manifests["validation"]["patients"],
    }
    with np.load(root / "fit_labels.npz") as a, np.load(root / "validation_labels.npz") as b:
        arrays = {k: (a[k] if k == "target_ids" else np.concatenate([a[k], b[k]])) for k in a.files}
    npz(root / "outer_labels.npz", **arrays)
    outer["labels"] = {"path": "outer_labels.npz", "sha256": sha(root / "outer_labels.npz")}
    write(root / "outer.json", outer)
    config = {
        "model": {
            "input_dim": 6,
            "hidden_dim": 8,
            "latent_dim": 4,
            "n_targets": 3,
            "n_cancers": 2,
            "adapter_rank": 2,
            "program_count": 2,
            "program_hidden": 5,
            "program_score_bound": 6.0,
            "aux_weight": 0.05,
        },
        "training": {
            "seed": 2025,
            "alpha_grid": [1.0, 10.0],
            "max_epochs": 2,
            "min_epochs": 1,
            "patience": 2,
            "lr": 1e-4,
            "linear_lr": 2e-5,
            "weight_decay": 1e-4,
            "batch_patients": 8,
            "chunk_size": 7,
            "gradient_clip": 1.0,
            "lr_patience": 1,
            "lr_factor": 0.5,
            "min_lr": 1e-6,
            "selection_mse_ratio_limit": 1.05,
            "selection_min_delta": 0.0001,
            "min_epochs_after_lr_reduction": 1,
            "min_lr_reductions": 1,
        },
    }
    write(root / "config.json", config)
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    make_demo(p.parse_args().output)
