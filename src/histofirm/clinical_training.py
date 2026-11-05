"""Fixed-epoch fitting of the adopted exploratory survival model."""

import numpy as np
from .survival import comparable


def fit_fixed(net, X, time, event, penalty, seed, epochs, adam):
    """Use a predetermined epoch count and penalty; do not tune on external data."""
    if int(np.sum(event)) < 10 or len(comparable(time, event)[0]) < 50:
        raise ValueError("Insufficient event or comparable-pair support.")
    theta = net.initialize(seed)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    curve = []
    for step in range(1, int(epochs) + 1):
        loss, grad = net.objective(theta, X, time, event, penalty, rank_weight=0.0)
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite clinical training objective.")
        norm = float(np.linalg.norm(grad))
        if norm > adam["clip"]:
            grad *= adam["clip"] / norm
        m = adam["beta1"] * m + (1 - adam["beta1"]) * grad
        v = adam["beta2"] * v + (1 - adam["beta2"]) * grad * grad
        mh = m / (1 - adam["beta1"] ** step)
        vh = v / (1 - adam["beta2"] ** step)
        theta -= adam["lr"] * mh / (np.sqrt(vh) + adam["eps"])
        curve.append(
            {"epoch": step, "pre_update_loss": float(loss), "pre_clip_gradient_norm": norm}
        )
    return theta, curve
