"""RSSMAdapter: bridges r2dreamer's tuple-based RSSM to SAILOR's dict-based interface.

r2dreamer RSSM uses (stoch, deter) tuples; SAILOR expects state dicts with
keys {"stoch", "deter", "logit"}.  This adapter translates between the two
so that SAILOR's MPPI planner, value training, and static_scan work unchanged.
"""

import torch
from torch import nn

from .r2d import RSSM


class RSSMAdapter(nn.Module):
    """Wraps ``r2dreamer.RSSM`` to present SAILOR's dict-based state interface."""

    def __init__(self, rssm: RSSM):
        super().__init__()
        self._rssm = rssm
        # Expose key dimensions for external consumers (ImagBehavior, etc.)
        self._stoch = rssm._stoch
        self._discrete = rssm._discrete
        self._deter = rssm._deter
        self._device = rssm._device
        self.feat_size = rssm.feat_size
        self.flat_stoch = rssm.flat_stoch

    # -- helpers ---------------------------------------------------------------

    def _pack(self, stoch, deter, logit=None):
        """Pack r2dreamer tensors into a SAILOR state dict."""
        state = {"stoch": stoch, "deter": deter}
        if logit is not None:
            state["logit"] = logit
        else:
            state["logit"] = torch.zeros(
                *stoch.shape[:-1], self._discrete, device=stoch.device
            )
        return state

    @staticmethod
    def _unpack(state):
        return state["stoch"], state["deter"]

    def _strip_pred_horizon(self, action):
        """Remove pred_horizon dim if present: (B, A, H) -> (B, A)."""
        if action.dim() >= 3 and action.shape[-1] != action.shape[-2]:
            return action[..., 0]
        return action

    # -- public interface (SAILOR-compatible) ----------------------------------

    def initial(self, batch_size):
        stoch, deter = self._rssm.initial(batch_size)
        logit = torch.zeros(
            batch_size, self._stoch, self._discrete, device=self._device
        )
        return {"stoch": stoch, "deter": deter, "logit": logit}

    def observe(self, embed, action, is_first, state=None):
        """Posterior rollout.

        Args:
            embed: (B, T, E) encoder embeddings.
            action: (B, T, A) or (B, T, A, H) actions.
            is_first: (B, T) episode-start flags.
            state: optional previous state dict (unused start -> zeros).

        Returns:
            (post, prior) — each a dict with (B, T, ...) tensors.
        """
        if state is not None:
            initial = self._unpack(state)
        else:
            initial = self._rssm.initial(embed.shape[0])

        # Normalise action to (B, T, A)
        act = action
        if act.dim() == 4:
            act = act[..., 0]

        reset = is_first  # same semantics

        stochs, deters, logits = self._rssm.observe(embed, act, initial, reset)

        # Build prior by running the prior network on each deterministic state
        B, T = deters.shape[:2]
        deters_flat = deters.reshape(B * T, -1)
        _, prior_logits_flat = self._rssm.prior(deters_flat)
        prior_logits = prior_logits_flat.reshape(B, T, self._stoch, self._discrete)
        prior_stochs = self._rssm.get_dist(prior_logits).rsample()

        post = {"stoch": stochs, "deter": deters, "logit": logits}
        prior = {"stoch": prior_stochs, "deter": deters, "logit": prior_logits}
        return post, prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Single posterior step.

        Args:
            prev_state: state dict or None (will use zeros).
            prev_action: (B, A) or (B, A, H) — previous action.
            embed: (B, E) encoder embedding.
            is_first: (B,) or (B, 1) episode-start flag.

        Returns:
            (post, prior) state dicts.
        """
        if prev_state is None:
            B = embed.shape[0]
            prev_state = self.initial(B)
            prev_action = torch.zeros(
                B, self._rssm._act_dim, device=self._device
            )

        stoch, deter = self._unpack(prev_state)
        act = self._strip_pred_horizon(prev_action)

        reset = is_first.squeeze(-1) if is_first.dim() > 1 else is_first

        new_stoch, new_deter, logit = self._rssm.obs_step(
            stoch, deter, act, embed, reset
        )
        _, prior_logit = self._rssm.prior(new_deter)
        prior_stoch = self._rssm.get_dist(prior_logit).rsample()

        post = {"stoch": new_stoch, "deter": new_deter, "logit": logit}
        prior = {"stoch": prior_stoch, "deter": new_deter, "logit": prior_logit}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """Single prior step (no observation).

        Args:
            prev_state: state dict.
            prev_action: (B, A) or (B, A, H).

        Returns:
            new state dict.
        """
        stoch, deter = self._unpack(prev_state)
        act = self._strip_pred_horizon(prev_action)
        new_stoch, new_deter = self._rssm.img_step(stoch, deter, act)
        _, logit = self._rssm.prior(new_deter)
        return {"stoch": new_stoch, "deter": new_deter, "logit": logit}

    def get_feat(self, state):
        """Extract flat feature vector from state dict."""
        return self._rssm.get_feat(state["stoch"], state["deter"])

    def get_dist(self, state):
        """Get distribution object from state dict."""
        return self._rssm.get_dist(state["logit"])

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """Compute KL losses, returning SAILOR's 4-tuple."""
        dyn_loss, rep_loss = self._rssm.kl_loss(
            post["logit"], prior["logit"], free
        )
        kl_value = dyn_loss + rep_loss
        kl_loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return kl_loss, kl_value, dyn_loss, rep_loss

    # -- passthrough -----------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use observe / obs_step / img_step directly.")
