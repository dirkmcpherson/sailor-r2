"""Distributional critic ensemble using r2dreamer's symexp_twohot MLPHead.

Drop-in replacement for SAILOR's ``MLPEnsemble`` but with distributional
(255-bin symexp_twohot) value heads and r2dreamer's LaProp + AGC optimiser.
"""

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from r2dreamer import MLPHead


class DistributionalCriticEnsemble(nn.Module):
    """Ensemble of distributional value function heads.

    Each member is an ``r2dreamer.MLPHead`` configured with ``symexp_twohot``
    distribution (255 bins).  The ensemble provides:

    - ``forward(feat)`` — returns the mean distribution across a random
      sub-sample of members (same interface as SAILOR's ``MLPEnsemble``).
    - ``get_std(feat)`` — inter-member standard deviation used by MPPI
      for uncertainty cost.
    - ``update(feat, target)`` — per-member negative log-prob loss.
    """

    def __init__(
        self,
        num_models,
        num_subsample,
        feat_size,
        layers=2,
        units=512,
        act="SiLU",
        device="cuda",
        dist="symexp_twohot",
    ):
        super().__init__()
        self.num_subsample = num_subsample
        assert 1 <= num_subsample <= num_models

        head_cfg = OmegaConf.create({
            "shape": [255],
            "layers": layers,
            "units": units,
            "act": act,
            "norm": True,
            "dist": {"name": dist, "bin_num": 255},
            "outscale": 0.0,
            "device": device,
            "symlog_inputs": False,
            "name": "value",
        })

        self.models = nn.ModuleList([
            MLPHead(head_cfg, feat_size) for _ in range(num_models)
        ])

    # -- Forward (SAILOR-compatible) -------------------------------------------

    @torch.no_grad()
    def forward(self, features):
        """Return distribution whose mode is the mean of sub-sampled members."""
        idxs = np.random.choice(len(self.models), self.num_subsample, replace=False)
        modes = torch.stack([self.models[i](features).mode() for i in idxs])
        mean_mode = modes.mean(dim=0)
        return _ScalarDist(mean_mode)

    def get_all_critic_mean(self, features):
        modes = torch.stack([m(features).mode() for m in self.models])
        return _ScalarDist(modes.mean(dim=0))

    # -- Update ----------------------------------------------------------------

    def update(self, features, target):
        """Compute per-model loss. Returns (T, B, 1) mean loss across models."""
        all_losses = torch.stack([
            -model(features).log_prob(target.detach()) for model in self.models
        ])  # (num_models, T, B)
        return all_losses.mean(dim=0)

    # -- Uncertainty -----------------------------------------------------------

    @torch.no_grad()
    def get_std(self, features):
        """Inter-member standard deviation of mode values."""
        all_modes = torch.stack([m(features).mode() for m in self.models])
        if all_modes.shape[0] <= 1:
            return torch.zeros_like(all_modes[0])
        return torch.std(all_modes, dim=0, unbiased=False)

    # -- Stats -----------------------------------------------------------------

    @torch.no_grad()
    def get_stats(self, features):
        all_modes = torch.stack([m(features).mode() for m in self.models])
        if all_modes.shape[0] <= 1:
            std = torch.zeros_like(all_modes[0])
        else:
            std = torch.std(all_modes, dim=0, unbiased=False)
        return {
            "ensemble_std_avg": std.mean().item(),
            "ensemble_std_max": std.max().item(),
            "ensemble_std_min": std.min().item(),
        }


class _ScalarDist:
    """Minimal distribution wrapper so ``.mode()`` and ``.log_prob()`` work."""

    def __init__(self, mode_val):
        self._mode = mode_val

    def mode(self):
        return self._mode

    def log_prob(self, target):
        return -torch.nn.functional.mse_loss(
            self._mode, target, reduction="none"
        ).sum(-1)
