"""Complete-patient, deterministic two-pass VJP for latent and tied-field moments."""

from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def moments(model, chunk_factory, device, cancer=None):
    n = 0
    count = 0
    mu = M2 = hx = ex = None
    for block in chunk_factory():
        a = block["x"] if isinstance(block, dict) else block
        x = torch.as_tensor(np.array(a, copy=True), dtype=model.x_center.dtype, device=device)
        if x.ndim != 2 or len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError("INVALID_PATCH_CHUNK")
        z0 = model.encode(x)
        e = model.extra_features(z0, x, cancer).double()
        z = z0.double()
        hn = model.normalize(x).double()
        m = len(z)
        bm = z.mean(0)
        bM = (z - bm).square().sum(0)
        if mu is None:
            mu = bm
            M2 = bM
            hx = hn.sum(0)
            ex = e.sum(0)
        else:
            delta = bm - mu
            total = n + m
            M2 = M2 + bM + delta.square() * (n * m / total)
            mu = mu + delta * (m / total)
            hx += hn.sum(0)
            ex += e.sum(0)
        n += m
        count += 1
    if n < 1:
        raise ValueError("EMPTY_PATIENT")
    return {
        "hmean": hx / n,
        "mean": mu,
        "second": M2 / n + mu.square(),
        "extra": ex / n,
        "n": n,
        "chunks": count,
        "cancer": cancer,
    }


def backward_encoder(model, chunk_factory, n, gmean, gsecond, device, gextra=None, cancer=None):
    seen = 0
    for block in chunk_factory():
        a = block["x"] if isinstance(block, dict) else block
        x = torch.as_tensor(np.array(a, copy=True), dtype=model.x_center.dtype, device=device)
        z = model.encode(x)
        grad = (gmean + 2 * z.detach() * gsecond) / n
        if model.extra_dim:
            e = model.extra_features(z, x, cancer)
            if gextra is None:
                raise ValueError("EXTRA_VJP_REQUIRED")
            # e may depend on raw/mean readout parameters as well as the encoder.
            # These gradients must be added, never discarded or encoder-only filtered.
            torch.autograd.backward(
                (z, e), (grad.to(z.dtype), (gextra / n).expand_as(e).to(e.dtype))
            )
        else:
            torch.autograd.backward(z, grad_tensors=grad.to(z.dtype))
        seen += len(z)
    if seen != n:
        raise ValueError("SECOND_PASS_PATCH_COUNT_CHANGED")


@torch.no_grad()
def predict_dataset(model, data, device, chunk_size, progress=None, diagnostics=None):
    result = []
    first = []
    increments = []
    C = {c: i for i, c in enumerate(data.cancer_names)}
    for i, c in enumerate(data.cancers):
        m = moments(model, lambda i=i: data.chunks(i, chunk_size), device, C[c])
        dtype = model.x_center.dtype
        cc = torch.tensor([C[c]], device=device)
        hm = m["hmean"][None].to(dtype)
        mm = m["mean"][None].to(dtype)
        ss = m["second"][None].to(dtype)
        ee = m["extra"][None]
        p = model.forward_stats(hm, mm, ss, cc, ee)
        q = model.first_order_stats(hm, mm, cc)
        result.append(p[0].double().cpu().numpy())
        first.append(q[0].double().cpu().numpy())
        increments.append(float((p - q).square().mean().sqrt()))
        if progress is not None:
            progress(i + 1, len(data.ids))
    if diagnostics is not None:
        diagnostics.update(
            first_order_is_actual_shared_readout=True,
            mean_total_heterogeneity_rms_standardized=float(np.mean(increments)),
            no_ST_accuracy_claim=True,
            variant=model.variant,
        )
    return np.asarray(result)
