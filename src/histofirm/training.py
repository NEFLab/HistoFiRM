"""Inner selection and fixed-epoch outer refitting."""

import random
import os
from pathlib import Path
import numpy as np
import torch
from .common import seed_int, write, npz, save_torch, digest
from .linear import prepare_base
from .prepare import validate_cache
from .selection import selection
from .program_axis import program_indices
from .model import HistoFiRM
from .stream import moments, backward_encoder, predict_dataset


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def fit(data, valid, cache, protocol, output, fold, device="cpu", selected=None):
    """Select using inner validation, or refit using its fixed alphas and epoch count.

    Split manifests are supplied explicitly; this function never creates or tunes
    outer folds. Evaluation data must not be passed as fit or validation data.
    """
    cfg = protocol["model"]
    t = protocol["training"]
    if len(data.target_ids) != cfg["n_targets"] or len(data.cancer_names) != cfg["n_cancers"]:
        raise ValueError("Model and data axes differ.")
    if valid is None and selected is None:
        raise ValueError("Refitting requires an inner-selection receipt.")
    if valid is not None and selected is not None:
        raise ValueError("Selection and refitting are distinct operations.")
    if valid is not None:
        if set(data.ids) & set(valid.ids):
            raise ValueError("Fit and validation patients overlap.")
        if data.target_ids != valid.target_ids or data.cancer_names != valid.cancer_names:
            raise ValueError("Fit and validation axes differ.")
    validate_cache(data, cache)
    if valid is not None:
        validate_cache(valid, cache)
    phase = "select" if valid is not None else "refit"
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory for each fit.")
    output.mkdir(parents=True, exist_ok=True)
    seed = seed_int(int(t["seed"]), int(fold), "IPDM")
    configure(seed)
    base, report = prepare_base(
        data, valid, cache, t["alpha_grid"], None if selected is None else selected["base_alphas"]
    )
    npz(output / "base.npz", **base)
    write(output / "base_report.json", report)
    model = HistoFiRM(cfg)
    model.init_base(base)
    model.init_program_axis(program_indices(data.target_ids, cfg["program_count"]))
    model.to(device)
    params = [p for n, p in model.named_parameters() if n != "raw_delta.weight" and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": t["lr"]},
            {"params": [model.raw_delta.weight], "lr": t["linear_lr"]},
        ],
        weight_decay=t["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t["lr_factor"], patience=t["lr_patience"], min_lr=t["min_lr"]
    )
    C = {c: i for i, c in enumerate(data.cancer_names)}
    ct = np.array([C[c] for c in data.cancers])
    yt = (data.labels - base["y_center"][ct]) / base["y_scale"][ct]
    mask = np.isfinite(yt) & base["variable"][ct]
    if np.any(mask.sum(1) == 0):
        raise ValueError("A fit patient has no variable, observed target.")
    yt = np.where(mask, yt, 0).astype("float32")
    weights = np.array([len(ct) / (len(C) * np.sum(ct == c)) for c in ct], dtype="float32")
    best_epoch = 0
    bad = 0
    reductions = 0
    last_reduction = 0
    best = None if valid is None else report["validation"].copy()
    n_epochs = t["max_epochs"] if valid is not None else int(selected["best_epoch"])
    curve = []
    rates_trace = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if valid is not None:
        cv = np.array([C[c] for c in valid.cancers])
        yv = (valid.labels - base["y_center"][cv]) / base["y_scale"][cv]
        support = {c: base["variable"][k] for k, c in enumerate(data.cancer_names)}
    stop = "maximum_epochs" if valid is not None else "fixed_epoch_refit"
    for ep in range(1, n_epochs + 1):
        if selected is not None:
            for group, lr in zip(optimizer.param_groups, selected["learning_rates"][ep - 1]):
                group["lr"] = lr
        rates = [g["lr"] for g in optimizer.param_groups]
        model.train()
        order = np.random.default_rng(seed_int(seed, ep, "order")).permutation(len(data.ids))
        losses = []
        for offset in range(0, len(order), t["batch_patients"]):
            ix = order[offset : offset + t["batch_patients"]]
            optimizer.zero_grad(set_to_none=True)
            stats = [
                moments(model, lambda i=int(i): data.chunks(i, t["chunk_size"]), device, int(ct[i]))
                for i in ix
            ]
            hm = torch.stack([s["hmean"] for s in stats]).float()
            mm = torch.stack([s["mean"] for s in stats]).float().requires_grad_(True)
            ss = torch.stack([s["second"] for s in stats]).float().requires_grad_(True)
            ee = torch.stack([s["extra"] for s in stats]).double().requires_grad_(True)
            cc = torch.as_tensor(ct[ix], device=device)
            yy = torch.as_tensor(yt[ix], device=device)
            mk = torch.as_tensor(mask[ix], device=device)
            ww = torch.as_tensor(weights[ix], device=device)
            pred = model.forward_stats(hm, mm, ss, cc, ee)
            mse = ((pred - yy).square() * mk).sum(1) / mk.sum(1).clamp_min(1)
            supervised = (mse * ww).sum() / len(ix)
            loss = supervised + cfg["aux_weight"] * model.auxiliary_losses(
                ee, yy, mk, ww, hm, mm, cc
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            gm = mm.grad.detach() if mm.grad is not None else torch.zeros_like(mm)
            gs = ss.grad.detach() if ss.grad is not None else torch.zeros_like(ss)
            ge = ee.grad.detach() if ee.grad is not None else torch.zeros_like(ee)
            for k, i in enumerate(ix):
                backward_encoder(
                    model,
                    lambda i=int(i): data.chunks(i, t["chunk_size"]),
                    stats[k]["n"],
                    gm[k],
                    gs[k],
                    device,
                    ge[k],
                    int(ct[i]),
                )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient.")
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        row = {"epoch": ep, "training_loss": float(np.mean(losses)), "learning_rates": rates}
        if valid is not None:
            pv = predict_dataset(model, valid, device, t["chunk_size"])
            sc = selection(yv, pv, valid.cancers, support)
            row["validation"] = sc
            if (
                sc["mse"] <= report["validation"]["mse"] * t["selection_mse_ratio_limit"]
                and sc["pearson"] > best["pearson"] + t["selection_min_delta"]
            ):
                best = sc
                best_epoch = ep
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            scheduler.step(sc["pearson"])
            if optimizer.param_groups[0]["lr"] < rates[0] - 1e-15:
                reductions += 1
                last_reduction = ep
        else:
            best_epoch = n_epochs
        curve.append(row)
        rates_trace.append(rates)
        write(output / "curve.json", curve)
        if (
            valid is not None
            and ep >= t["min_epochs"]
            and bad >= t["patience"]
            and reductions >= t["min_lr_reductions"]
            and ep - last_reduction >= t["min_epochs_after_lr_reduction"]
        ):
            stop = "early_stop_after_low_lr_exposure"
            break
    if selected is not None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state = {
        "model_config": cfg,
        "variant": "AUX_ONLY",
        "state_dict": best_state,
        "target_ids": data.target_ids,
        "cancer_names": data.cancer_names,
        "fold": int(fold),
        "phase": phase,
        "best_epoch": best_epoch,
    }
    save_torch(output / "model.pt", state)
    receipt = {
        "phase": phase,
        "fold": int(fold),
        "base_alphas": report["alphas"],
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "stop_reason": stop,
        "learning_rates": rates_trace[:best_epoch]
        if selected is None
        else selected["learning_rates"],
        "best_validation": best,
        "protocol_sha256": digest(protocol),
        "target_ids": data.target_ids,
        "cancer_names": data.cancer_names,
    }
    write(output / "fit.json", receipt)
    return receipt
