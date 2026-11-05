"""HistoFiRM with training-only Hallmark auxiliary supervision."""

import torch
from torch import nn
from .core import MomentRegressor as Core


class HistoFiRM(Core):
    def __init__(self, cfg, variant="AUX_ONLY"):
        if variant != "AUX_ONLY":
            raise ValueError("This implementation exposes the adopted auxiliary-supervision model.")
        super().__init__(cfg, "MAIN")
        self.variant = variant
        self.extra_dim = cfg["program_count"]
        self.register_buffer("program_index", torch.zeros(cfg["program_count"], dtype=torch.int64))
        self.register_buffer("program_axis_ready", torch.tensor(False))
        self.program_network = nn.Sequential(
            nn.Linear(cfg["latent_dim"], cfg["program_hidden"]),
            nn.GELU(),
            nn.Linear(cfg["program_hidden"], cfg["program_count"]),
        )

    def init_program_axis(self, indices):
        x = torch.as_tensor(indices, dtype=torch.int64, device=self.program_index.device)
        if x.shape != self.program_index.shape or torch.unique(x).numel() != x.numel():
            raise ValueError("HALLMARK_AXIS_INVALID")
        if bool((x < 0).any()) or bool((x >= self.cfg["n_targets"]).any()):
            raise ValueError("HALLMARK_INDEX_OUT_OF_RANGE")
        self.program_index.copy_(x)
        self.program_axis_ready.fill_(True)

    def first_order_stats(self, hmean, mean, cancers):
        """E[first-order local field], in per-cancer TRAINING standardized units."""
        return Core.forward_stats(self, hmean, mean, mean.square(), cancers)

    def functional_field(self, x, cancer, z=None):
        """Actual first-order prediction field for every target; no separate map head."""
        c = int(cancer)
        if z is None:
            z = self.encode(x)
        L = self.cfg["latent_dim"]
        ad = self.adapter_A[c] @ self.adapter_B[c] / self.cfg["adapter_rank"] ** 0.5
        u = self.mean_head.weight.T + ad[:L]
        f = self.normalize(x) @ (self.base_coef[c] + self.raw_delta.weight.T)
        f = f + self.base_bias[c] + self.offset[c] + z @ u
        # Constant/untrained outputs are excluded from training; masks remain in decode.
        return torch.where(self.variable[c] & self.supported[c], f, torch.zeros_like(f))

    def program_scores(self, z):
        if self.variant != "AUX_ONLY":
            raise ValueError("ONLY_AUX_ONLY_HAS_PARALLEL_PROGRAM_HEAD")
        b = float(self.cfg["program_score_bound"])
        return b * torch.tanh(self.program_network(z) / b)

    def forward_bag(self, x, cancer):
        z = self.encode(x)
        ex = self.extra_features(z, x, cancer).mean(0, keepdim=True)
        cc = torch.as_tensor([cancer], device=x.device)
        return self.forward_stats(
            self.normalize(x).mean(0, keepdim=True),
            z.mean(0, keepdim=True),
            z.square().mean(0, keepdim=True),
            cc,
            ex,
        )[0]

    def extra_features(self, z, x=None, cancer=None):
        return self.program_scores(z)

    def forward_stats(self, hmean, mean, second, cancers, extra=None):
        return Core.forward_stats(self, hmean, mean, second, cancers)

    def auxiliary_losses(
        self, extra, labels_standardized, mask, patient_weights, hmean=None, mean=None, cancers=None
    ):
        if not bool(self.program_axis_ready):
            raise RuntimeError("Program axis is not initialized.")
        yy = labels_standardized[:, self.program_index]
        mm = mask[:, self.program_index]
        per = ((extra - yy).square() * mm).sum(1) / mm.sum(1).clamp_min(1)
        return (per * patient_weights).sum() / len(per)
