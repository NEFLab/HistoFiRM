# SPDX-License-Identifier: GPL-3.0-or-later
# Author HE2RNA core Copyright (C) 2020 Owkin Inc.; B3 interface adaptations 2026.
"""B3 HE2RNA adapter for explicit bags and patient-level supervision.

Author family: owkin/HE2RNA_code, commit
d47f481fbcaab41a49635e4a35e810bf30eae0e0 (GPL-3.0-or-later).
The author model.py SHA256 is recorded in SOURCE_PROVENANCE.json.

The default hidden layers, activation, dropout and k list are bound to the
author's config_all_genes.ini. Project changes are explicit bag masks (signed
UNI2H and short bags), input layout, and patch-count-weighted patient pooling.
No gene clipping, standardization, training loop or truth access occurs here.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn


AUTHOR_COMMIT = "d47f481fbcaab41a49635e4a35e810bf30eae0e0"
AUTHOR_MODEL_SHA256 = "1f3fdb5e4ac401b76dde1212b7d73c70e51c13ec9549a73e9e12319e82705fae"
AUTHOR_CONFIG_SHA256 = "76de43c9cd176311574373a821616f9a7c2527cc29675d614c724656d151329a"
AUTHOR_KS = (1, 2, 5, 10, 20, 50, 100)


class HE2RNAAdapter(nn.Module):
    """Shared per-tile network followed by independent per-gene top-k means.

    Input is [n_slides, n_tiles, input_dim]. Supply valid_lengths for contiguous
    prefix padding or valid_mask [n_slides, n_tiles] for arbitrary padding.
    If both are supplied, mask counts must equal valid_lengths.

    Training samples one k with NumPy per forward call, shared by all slides
    and genes in that call. Set random_k to a configured k for semantic tests
    or deterministic replay. Evaluation always averages all configured k's.
    Short bags use min(k, valid_count) independently for every slide; repeated
    effective k's still retain their original weight in the seven-k average.

    bias_init should be the per-gene mean of the *current fitting stage's*
    training targets, matching the author initialization. The caller owns
    missing-label handling and fitting-stage target standardization.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        output_dim: int = 18977,
        hidden: Sequence[int] = (1024, 1024),
        dropout: float = 0.25,
        ks: Sequence[int] = AUTHOR_KS,
        bias_init: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden, output_dim)
        if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims):
            raise ValueError("All network dimensions must be positive integers.")
        if not ks or any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0 for k in ks):
            raise ValueError("ks must contain positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = tuple(hidden)
        self.ks = tuple(int(k) for k in ks)
        # Keep the original conv0/conv1/conv2 registration and parameter names.
        self.layers: list[nn.Conv1d] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layer = nn.Conv1d(din, dout, kernel_size=1, stride=1, bias=True)
            setattr(self, "conv" + str(i), layer)
            self.layers.append(layer)
        self.nonlin = nn.ReLU()
        self.do = nn.Dropout(dropout)
        if bias_init is not None:
            bias = torch.as_tensor(bias_init, dtype=self.layers[-1].bias.dtype)
            if bias.shape != (output_dim,) or not bool(torch.isfinite(bias).all()):
                raise ValueError("bias_init must be a finite vector of output_dim.")
            with torch.no_grad():
                self.layers[-1].bias.copy_(bias)

    def conv(self, channel_first: Tensor) -> Tensor:
        """Author Conv -> ReLU -> Dropout hidden blocks, then linear output."""
        x = channel_first
        for layer in self.layers[:-1]:
            x = self.do(self.nonlin(layer(x)))
        return self.layers[-1](x)

    def _validity(
        self,
        x: Tensor,
        valid_lengths: Tensor | Sequence[int] | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("x must have shape [n_slides, n_tiles, input_dim].")
        n_slides, n_tiles, _ = x.shape
        if n_slides == 0 or n_tiles == 0:
            raise ValueError("Empty batches and empty bags are not valid.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")
        lengths = None
        if valid_lengths is not None:
            raw = torch.as_tensor(valid_lengths, device=x.device)
            if raw.shape != (n_slides,) or raw.dtype == torch.bool:
                raise ValueError("valid_lengths must contain one integer per slide.")
            if raw.is_floating_point() and (not bool(torch.isfinite(raw).all()) or not bool((raw == raw.floor()).all())):
                raise ValueError("valid_lengths must contain finite integers.")
            lengths = raw.to(dtype=torch.long)
            if bool(((lengths < 1) | (lengths > n_tiles)).any()):
                raise ValueError("Every valid length must be in [1, n_tiles].")
        if valid_mask is not None:
            mask = torch.as_tensor(valid_mask, device=x.device)
            if mask.shape != (n_slides, n_tiles) or mask.dtype != torch.bool:
                raise ValueError("valid_mask must be boolean [n_slides, n_tiles].")
            counts = mask.sum(dim=1)
            if bool((counts == 0).any()):
                raise ValueError("Every slide must have at least one valid tile.")
            if lengths is not None and not torch.equal(counts, lengths):
                raise ValueError("valid_lengths disagree with valid_mask counts.")
        elif lengths is not None:
            mask = torch.arange(n_tiles, device=x.device)[None, :] < lengths[:, None]
            counts = lengths
        else:
            raise ValueError("Provide explicit valid_lengths or valid_mask.")
        return mask, counts

    def _tile_scores(self, x: Tensor, mask: Tensor) -> Tensor:
        # Neutralize even non-finite padding before the network; padding has
        # neither a score nor an input/parameter-gradient contribution.
        clean = torch.where(mask[..., None], x, torch.zeros((), dtype=x.dtype, device=x.device))
        scores = self.conv(clean.transpose(1, 2))
        return scores.masked_fill(~mask[:, None, :], -torch.inf)

    @staticmethod
    def _pool(scores: Tensor, counts: Tensor, k: int) -> Tensor:
        width = min(int(k), scores.shape[-1])
        top_values = torch.topk(scores, width, dim=2, largest=True, sorted=True).values
        effective_k = counts.clamp(max=width)
        selected = torch.arange(width, device=scores.device)[None, None, :] < effective_k[:, None, None]
        # Use where, not multiplication: padding -inf times zero would be NaN.
        finite_selected = torch.where(selected, top_values, torch.zeros((), dtype=scores.dtype, device=scores.device))
        return finite_selected.sum(dim=2) / effective_k[:, None].to(scores.dtype)

    def forward(
        self,
        x: Tensor,
        valid_lengths: Tensor | Sequence[int] | None = None,
        *,
        valid_mask: Tensor | None = None,
        random_k: int | None = None,
    ) -> Tensor:
        mask, counts = self._validity(x, valid_lengths, valid_mask)
        if self.training:
            k = int(np.random.choice(self.ks)) if random_k is None else random_k
            if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k not in self.ks:
                raise ValueError("random_k must be one of the configured k values.")
            scores = self._tile_scores(x, mask)
            return self._pool(scores, counts, int(k))
        if random_k is not None:
            raise ValueError("random_k is a training-only override; evaluation averages all k.")
        scores = self._tile_scores(x, mask)
        result = torch.zeros(scores.shape[:2], dtype=scores.dtype, device=scores.device)
        for k in self.ks:
            result = result + self._pool(scores, counts, k) / len(self.ks)
        return result

    def forward_patients(
        self,
        x: Tensor,
        valid_lengths: Tensor | Sequence[int] | None,
        slide_to_patient: Tensor | Sequence[int],
        patch_counts: Tensor | Sequence[int],
        *,
        valid_mask: Tensor | None = None,
        random_k: int | None = None,
        num_patients: int | None = None,
    ) -> Tensor:
        """Predict slides first; aggregate using original, pre-clustering counts.

        slide_to_patient is a zero-based contiguous patient index. patch_counts
        contains each slide's original patch count, *not* its supertile count.
        """
        slide_predictions = self.forward(
            x, valid_lengths, valid_mask=valid_mask, random_k=random_k
        )
        return aggregate_slide_predictions(
            slide_predictions, slide_to_patient, patch_counts,
            num_patients=num_patients,
        )


def aggregate_slide_predictions(
    slide_predictions: Tensor,
    slide_to_patient: Tensor | Sequence[int],
    patch_counts: Tensor | Sequence[int],
    *,
    num_patients: int | None = None,
) -> Tensor:
    """Differentiable original-patch-count-weighted mean of slide predictions."""
    if slide_predictions.ndim != 2 or slide_predictions.shape[0] == 0:
        raise ValueError("slide_predictions must be non-empty [n_slides, n_genes].")
    n_slides = slide_predictions.shape[0]
    device = slide_predictions.device
    mapping_raw = torch.as_tensor(slide_to_patient, device=device)
    if mapping_raw.shape != (n_slides,) or mapping_raw.dtype == torch.bool:
        raise ValueError("slide_to_patient must contain one integer per slide.")
    if mapping_raw.is_floating_point() and (not bool(torch.isfinite(mapping_raw).all()) or not bool((mapping_raw == mapping_raw.floor()).all())):
        raise ValueError("slide_to_patient must contain finite integers.")
    mapping = mapping_raw.to(torch.long)
    if bool((mapping < 0).any()):
        raise ValueError("Patient indices must be nonnegative.")
    required_patients = int(mapping.max().item()) + 1
    if num_patients is None:
        num_patients = required_patients
    if not isinstance(num_patients, int) or isinstance(num_patients, bool) or num_patients < required_patients:
        raise ValueError("num_patients must cover all mapped patients.")
    counts = torch.as_tensor(patch_counts, dtype=slide_predictions.dtype, device=device)
    if counts.shape != (n_slides,) or not bool(torch.isfinite(counts).all()) or bool((counts <= 0).any()) or not bool((counts == counts.floor()).all()):
        raise ValueError("patch_counts must contain one positive integer per slide.")
    denominator = slide_predictions.new_zeros(num_patients)
    denominator = denominator.index_add(0, mapping, counts)
    if bool((denominator == 0).any()):
        raise ValueError("Each patient index must have at least one mapped slide.")
    numerator = slide_predictions.new_zeros((num_patients, slide_predictions.shape[1]))
    numerator = numerator.index_add(0, mapping, slide_predictions * counts[:, None])
    return numerator / denominator[:, None]
