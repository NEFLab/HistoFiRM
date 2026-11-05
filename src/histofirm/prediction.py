"""Patient prediction, fold aggregation and computational local readouts."""

import numpy as np
import torch
from .model import HistoFiRM
from .stream import moments, predict_dataset
from .common import npz


def load_model(checkpoint, device="cpu"):
    # Load only checkpoints obtained from a trusted source.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = HistoFiRM(state["model_config"], state["variant"]).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval(), state


@torch.no_grad()
def predict(checkpoint, data, output, device="cpu", chunk_size=1024):
    model, state = load_model(checkpoint, device)
    if data.target_ids != state["target_ids"] or data.cancer_names != state["cancer_names"]:
        raise ValueError("Checkpoint and manifest axes differ.")
    standardized = predict_dataset(model, data, device, chunk_size)
    c = torch.tensor([data.cancer_names.index(x) for x in data.cancers], device=device)
    result = (
        model.decode(torch.as_tensor(standardized, device=device, dtype=model.x_center.dtype), c)
        .cpu()
        .numpy()
    )
    npz(
        output,
        prediction=result,
        patient_ids=np.asarray(data.ids),
        cancers=np.asarray(data.cancers),
        target_ids=np.asarray(data.target_ids),
        fold=np.asarray(state["fold"]),
    )
    return result


def merge_predictions(files, output, external=False):
    arrays = []
    for filename in files:
        with np.load(filename, allow_pickle=False) as z:
            arrays.append({k: z[k].copy() for k in z.files})
    if len(arrays) != 5 or len({int(z["fold"]) for z in arrays}) != 5:
        raise ValueError("Exactly five distinct fold outputs are required.")
    first = arrays[0]
    if any(not np.array_equal(z["target_ids"], first["target_ids"]) for z in arrays):
        raise ValueError("Target axes differ.")
    if external:
        if any(
            not np.array_equal(z[k], first[k]) for z in arrays for k in ("patient_ids", "cancers")
        ):
            raise ValueError("External patient axes differ.")
        result = {k: first[k] for k in ("patient_ids", "cancers", "target_ids")}
        result["prediction"] = np.mean([z["prediction"] for z in arrays], axis=0)
    else:
        result = {
            k: np.concatenate([z[k] for z in arrays], axis=0)
            for k in ("patient_ids", "cancers", "prediction")
        }
        result["target_ids"] = first["target_ids"]
        if len(set(result["patient_ids"])) != len(result["patient_ids"]):
            raise ValueError("Held-out patient IDs overlap.")
    npz(output, **result)
    return result


@torch.no_grad()
def local_readout(model, chunk_factory, cancer, device="cpu"):
    """Return first-order response and full-bag heterogeneity contribution.

    Their patchwise sum, averaged over the complete bag, reconstructs the patient
    output. Neither component is an attention weight or measured local RNA.
    """
    st = moments(model, chunk_factory, device, cancer)
    dtype = model.x_center.dtype
    response = []
    heterogeneity = []
    for block in chunk_factory():
        a = block["x"] if isinstance(block, dict) else block
        x = torch.as_tensor(np.array(a, copy=True), dtype=dtype, device=device)
        r, h = model.local_components(x, cancer, st["mean"].to(dtype))
        response.append(r.cpu().numpy())
        heterogeneity.append(h.cpu().numpy())
    return np.concatenate(response), np.concatenate(heterogeneity)
