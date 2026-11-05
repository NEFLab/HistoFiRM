"""Ridge-anchored regression using complete-bag feature moments."""

from __future__ import annotations
import torch
from torch import nn


class MomentRegressor(nn.Module):
    def __init__(self, cfg, variant="MAIN"):
        super().__init__()
        self.cfg = dict(cfg)
        self.variant = variant
        D = cfg["input_dim"]
        H = cfg["hidden_dim"]
        L = cfg["latent_dim"]
        T = cfg["n_targets"]
        C = cfg["n_cancers"]
        R = cfg["adapter_rank"]
        self.stem = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, L), nn.GELU())
        self.residual = nn.Sequential(nn.Linear(L, H), nn.GELU(), nn.Linear(H, L))
        self.raw_delta = nn.Linear(D, T, bias=False)
        self.mean_head = nn.Linear(L, T, bias=False)
        self.var_head = nn.Linear(L, T, bias=False)
        self.adapter_A = nn.Parameter(torch.empty(C, 2 * L, R))
        self.adapter_B = nn.Parameter(torch.zeros(C, R, T))
        nn.init.normal_(self.adapter_A, std=0.02)
        self.offset = nn.Parameter(torch.zeros(C, T))
        for m in (self.raw_delta, self.mean_head, self.var_head):
            nn.init.zeros_(m.weight)
        self.register_buffer("x_center", torch.zeros(D))
        self.register_buffer("x_scale", torch.ones(D))
        self.register_buffer("y_center", torch.zeros(C, T))
        self.register_buffer("y_scale", torch.ones(C, T))
        self.register_buffer("base_coef", torch.zeros(C, D, T))
        self.register_buffer("base_bias", torch.zeros(C, T))
        self.register_buffer("supported", torch.ones(C, T, dtype=torch.bool))
        self.register_buffer("variable", torch.ones(C, T, dtype=torch.bool))
        if variant == "MEAN":
            self.var_head.weight.requires_grad_(False)
        elif variant == "RAW_ADAPT":
            for name, param in self.named_parameters():
                if name not in ("raw_delta.weight", "offset"):
                    param.requires_grad_(False)
        elif variant != "MAIN":
            raise ValueError("UNKNOWN_VARIANT")

    def init_base(self, arrays):
        for name in [
            "x_center",
            "x_scale",
            "y_center",
            "y_scale",
            "base_coef",
            "base_bias",
            "supported",
            "variable",
        ]:
            t = torch.as_tensor(
                arrays[name], dtype=getattr(self, name).dtype, device=getattr(self, name).device
            )
            if t.shape != getattr(self, name).shape:
                raise ValueError("BASE_SHAPE " + name)
            getattr(self, name).copy_(t)

    def normalize(self, x):
        return (x - self.x_center) / self.x_scale

    def encode(self, x):
        z = self.stem(self.normalize(x))
        return z + 0.25 * self.residual(z)

    def forward_stats(self, hmean, mean, second, cancers):
        """hmean is mean normalized raw h; mean/second are full-bag E[z], E[z^2]."""
        c = cancers.long()
        v = (second - mean.square()).clamp_min(0)
        if self.variant in ("MEAN", "RAW_ADAPT"):
            v = torch.zeros_like(v)
        if self.variant == "RAW_ADAPT":
            mean = torch.zeros_like(mean)
        q = torch.cat([mean, v], dim=1)
        fixed = torch.einsum("bd,bdt->bt", hmean, self.base_coef[c]) + self.base_bias[c]
        adaptive = torch.einsum("bl,blr->br", q, self.adapter_A[c])
        adaptive = torch.einsum("br,brt->bt", adaptive, self.adapter_B[c]) / (
            self.cfg["adapter_rank"] ** 0.5
        )
        return (
            fixed
            + self.raw_delta(hmean)
            + self.mean_head(mean)
            + self.var_head(v)
            + adaptive
            + self.offset[c]
        )

    def forward_bag(self, x, cancer):
        z = self.encode(x)
        h = self.normalize(x).mean(0, keepdim=True)
        return self.forward_stats(
            h,
            z.mean(0, keepdim=True),
            z.square().mean(0, keepdim=True),
            torch.as_tensor([cancer], device=x.device),
        )[0]

    def decode(self, p, cancers):
        c = cancers.long()
        v = p * self.y_scale[c] + self.y_center[c]
        v = torch.where(self.variable[c], v, self.y_center[c])
        return torch.where(self.supported[c], v, torch.full_like(v, float("nan")))

    def local_components(self, x, cancer, full_mean):
        """Returns first-order response and bag-conditional heterogeneity contribution.
        Means of these two fields reconstruct patient prediction; neither is ST ground truth.
        """
        c = int(cancer)
        z = self.encode(x)
        xn = self.normalize(x)
        L = self.cfg["latent_dim"]
        ad = self.adapter_A[c] @ self.adapter_B[c] / self.cfg["adapter_rank"] ** 0.5
        u = self.mean_head.weight.T + ad[:L]
        v = self.var_head.weight.T + ad[L:]
        if self.variant == "MEAN":
            v = v * 0
        if self.variant == "RAW_ADAPT":
            u = u * 0
            v = v * 0
        loc = (
            xn @ (self.base_coef[c] + self.raw_delta.weight.T)
            + self.base_bias[c]
            + self.offset[c]
            + z @ u
        )
        heter = (z - full_mean).square() @ v
        response = loc * self.y_scale[c] + self.y_center[c]
        heter = heter * self.y_scale[c]
        response = torch.where(self.variable[c], response, self.y_center[c])
        heter = torch.where(self.variable[c], heter, torch.zeros_like(heter))
        response = torch.where(self.supported[c], response, torch.full_like(response, float("nan")))
        heter = torch.where(self.supported[c], heter, torch.full_like(heter, float("nan")))
        return response, heter

    def parameter_counts(self):
        return {
            "trainable_neural": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "all_neural": sum(p.numel() for p in self.parameters()),
            "fixed_ridge_coefficients": self.base_coef.numel() + self.base_bias.numel(),
            "fixed_statistics_elements": sum(
                b.numel() for n, b in self.named_buffers() if not n.startswith("base_")
            ),
        }
