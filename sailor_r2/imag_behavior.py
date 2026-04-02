"""ImagBehavior: MPPI planner and value training, adapted for r2dreamer world model.

Based on SAILOR's ``imag_behavior.py`` but uses:
- ``DistributionalCriticEnsemble`` (symexp_twohot) instead of ``MLPEnsemble``
- r2dreamer's ``LaProp`` + ``clip_grad_agc_`` for the value optimiser
- ``RSSMAdapter`` (dict-based interface, same as original SAILOR)
"""

import copy

import einops
import torch
import torch.nn as nn

from sailor.dreamer import tools

from .critic import DistributionalCriticEnsemble

_to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """Running quantile-based reward normalisation."""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, base_policy):
        super().__init__()
        self._use_amp = config.precision == 16
        self._config = config
        self._world_model = world_model
        self.base_policy = base_policy
        self.reliability_fn = None
        self.cost_fn = None

        feat_size = world_model.dynamics.feat_size

        # Distributional critic ensemble (r2dreamer symexp_twohot)
        self.value = DistributionalCriticEnsemble(
            num_models=config.critic["num_models"],
            num_subsample=config.critic["num_subsample"],
            feat_size=feat_size,
            layers=config.critic["layers"],
            units=int(getattr(config, "units", 512)),
            act=str(getattr(config, "act", "SiLU")),
            device=str(config.device),
        )
        if config.critic.get("slow_target", True):
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        # Value optimiser (SAILOR's Optimizer wrapping Adam + grad clip)
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"Value ensemble: {sum(p.numel() for p in self.value.parameters()):,} params")

        if getattr(config, "reward_EMA", True):
            self.register_buffer("ema_vals", torch.zeros(2, device=config.device))
            self.reward_ema = RewardEMA(device=config.device)

    # -- Training (value function on imagined rollouts) ------------------------

    def _train(self, input_data, objective, training_step):
        self._update_slow_target()
        metrics = {}

        imag_feat, imag_state, imag_action_dict = self._imagine(
            input_data, self._config.imag_horizon, mode="base_only"
        )

        reward = objective(imag_feat, imag_state, imag_action_dict["base_action"])
        target, weights, _ = self._compute_target(imag_feat, imag_state, reward)

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                value = self.value(imag_feat[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = self.value.update(imag_feat[:-1].detach(), target)
                if self._config.critic.get("slow_target", True):
                    slow_target = self._slow_value(imag_feat[:-1].detach())
                    slow_loss = self.value.update(
                        imag_feat[:-1].detach(), slow_target.mode().detach()
                    )
                    metrics.update(tools.tensorstats(value_loss, "orig_critic_loss"))
                    metrics.update(tools.tensorstats(slow_loss, "slow_critic_loss"))
                    value_loss = value_loss + slow_loss
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(imag_action_dict["base_action"], "imag_base_action"))
        metrics.update(self.value.get_stats(features=imag_feat[:-1].detach()))

        with tools.RequiresGrad(self):
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, None, weights, metrics

    # -- Imagination rollout ---------------------------------------------------

    def _imagine(self, input_data, horizon, mode="base_only"):
        assert mode in ("default", "residual_buffer", "base_only")
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in input_data["post"].items()}
        start["ID"] = torch.tensor(0)

        base_action = flatten(input_data["obs_orig"]["base_action"])[..., :horizon]
        residual_action = flatten(input_data["obs_orig"]["residual_action"])[..., :horizon]

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            base_policy_action = base_action[..., state["ID"]]
            if mode == "base_only":
                residual_policy_action = torch.zeros_like(base_policy_action)
            elif mode == "residual_buffer":
                residual_policy_action = residual_action[..., state["ID"]]
            else:
                raise NotImplementedError(mode)
            action_dict = {
                "base_action": base_policy_action,
                "residual_action": residual_policy_action,
            }
            action_sum = self.get_action_sum(base_policy_action, residual_policy_action)
            succ = dynamics.img_step(state, action_sum)
            succ["ID"] = state["ID"] + 1
            return succ, feat, action_dict

        succ, feats, actions_dict = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions_dict

    # -- Action sum (straight-through gradient trick) --------------------------

    def get_action_sum(self, base_actions, residual_actions):
        assert base_actions.shape == residual_actions.shape
        return (
            (base_actions + residual_actions).clamp(-1, 1).detach()
            + residual_actions
            - residual_actions.detach()
        )

    # -- MPPI planner ----------------------------------------------------------

    def reset(self):
        self.base_policy.reset()

    def get_action(self, obs, feat, latent, weighting_in_base=True):
        with torch.no_grad():
            base_action = self.base_policy.get_action(
                obs, weighting=weighting_in_base, get_full_action=True
            )
            base_action = torch.tensor(base_action, device=feat.device, dtype=feat.dtype)

            if self.reliability_fn is not None and hasattr(self.reliability_fn, "set_initial_ee_pos"):
                import numpy as np
                state = obs["state"]
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, device=feat.device, dtype=feat.dtype)
                ee_pos = state[..., 9:12]
                if ee_pos.dim() == 1:
                    ee_pos = ee_pos.unsqueeze(0)
                self.reliability_fn.set_initial_ee_pos(ee_pos)

            mppi_action = self.mppi_actions(latent, base_action=base_action)
            action_dict = {
                "base_action": base_action[:, :, 0],
                "residual_action": mppi_action[:, :, 0],
            }
        return action_dict

    def mppi_actions(self, latent, base_action):
        num_samples = self._config.mppi["num_samples"]
        horizon = self._config.mppi["horizon"]
        action_dim = self._config.num_actions
        _BS = latent["stoch"].shape[0]

        _state = {k: v.unsqueeze(1) for k, v in latent.items()}
        _state = {
            k: v.repeat(*[1 if i != 1 else num_samples for i in range(len(v.shape))])
            for k, v in _state.items()
        }
        base_action = base_action.unsqueeze(1).expand(-1, num_samples, -1, -1)

        mean = torch.zeros(_BS, action_dim, horizon, device=self._config.device)
        std = torch.ones(_BS, action_dim, horizon, device=self._config.device) * self._config.mppi["init_std"]

        for _ in range(self._config.mppi["iterations"]):
            mppi_actions = (
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(_BS, action_dim, num_samples, horizon, device=self._config.device)
            ).clamp(-self._config.mppi["abs_residual"], self._config.mppi["abs_residual"])

            input_data = {
                "post": _state,
                "obs_orig": {
                    "base_action": base_action,
                    "residual_action": mppi_actions.permute(0, 2, 1, 3),
                },
            }
            imag_feat, imag_state, imag_action = self._imagine(input_data, horizon, mode="residual_buffer")

            with torch.no_grad():
                if self.cost_fn is not None:
                    final_value = self.cost_fn(
                        imag_feat=imag_feat, imag_state=imag_state,
                        imag_action=imag_action, world_model=self._world_model,
                        value_fn=self.value, config=self._config,
                        reliability_fn=self.reliability_fn,
                    )
                else:
                    if self._config.train_dp_mppi_params["use_discrim"]:
                        if self._config.train_dp_mppi_params["discrim_state_only"]:
                            get_reward = lambda f, a: self._world_model.get_reward(f).mode()
                        else:
                            get_reward = lambda f, a: self._world_model.get_reward(torch.cat([f, a], dim=-1)).mode()
                    else:
                        get_reward = lambda f, a: self._world_model.get_reward(f).mode()

                    get_cont = lambda f: self._world_model.heads["cont"](f).mean

                    reliability_mode = self._config.mppi.get("reliability_mode", "mask")
                    if self.reliability_fn is not None and reliability_mode == "mask":
                        unreliable_mask = torch.zeros(_BS * num_samples, dtype=torch.bool, device=self._config.device)

                    G, discount = 0, 1
                    for t in range(horizon - 1):
                        total_action = torch.clamp(
                            imag_action["base_action"][t] + imag_action["residual_action"][t], -1, 1
                        )
                        reward = get_reward(imag_feat[t], total_action)
                        cont = get_cont(imag_feat[t])
                        G += discount * reward

                        if self._config.mppi["uncertainty_cost"] > 0:
                            G -= discount * self._config.mppi["uncertainty_cost"] * self.value.get_std(imag_feat[t])
                        if self._config.mppi["action_l2_cost"] > 0:
                            act_norm = torch.norm(total_action, dim=-1)[:, None] / action_dim
                            G -= discount * self._config.mppi["action_l2_cost"] * act_norm

                        if self.reliability_fn is not None:
                            reliability = self.reliability_fn(imag_feat[t], imag_state, imag_action, t)
                            if reliability_mode == "mask":
                                unreliable_mask |= reliability < self._config.mppi.get("reliability_threshold", 0.5)
                            elif reliability_mode == "cost":
                                G -= discount * self._config.mppi.get("reliability_cost_weight", 1.0) * (1.0 - reliability).unsqueeze(-1)

                        discount *= self._config.mppi["discount"] * cont

                    final_value = G + discount * self.value(imag_feat[-1]).mode()
                    if self._config.mppi["uncertainty_cost"] > 0:
                        final_value -= discount * self._config.mppi["uncertainty_cost"] * self.value.get_std(imag_feat[-1])
                    if self.reliability_fn is not None and reliability_mode == "mask":
                        final_value[unreliable_mask] = -1e10

            values = final_value.squeeze(-1).reshape(_BS, num_samples)
            elite_idxs = torch.topk(values, self._config.mppi["num_elites"], dim=1).indices
            elite_actions = torch.gather(
                mppi_actions.permute(0, 3, 2, 1),
                2,
                elite_idxs.unsqueeze(1).unsqueeze(-1).expand(-1, horizon, -1, action_dim),
            )
            elite_value = torch.gather(values, 1, elite_idxs)

            max_value = elite_value.max(1)[0]
            score = torch.exp(self._config.mppi["temperature"] * (elite_value - max_value.unsqueeze(1)))
            score /= score.sum(1, keepdim=True)
            score = score.unsqueeze(1).expand(-1, horizon, -1).unsqueeze(-1)
            mean = (torch.sum(score * elite_actions, dim=2) / (score.sum(2, keepdim=True) + 1e-9).squeeze(-1))
            std = torch.sqrt(
                torch.sum(score * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2)
                / (score.sum(2, keepdim=True) + 1e-9).squeeze(-1)
            ).clamp_(self._config.mppi["min_std"], self._config.mppi["max_std"])
            mean = mean.permute(0, 2, 1)
            std = std.permute(0, 2, 1)

        max_value_idx = values.argmax(dim=1)
        expanded_idx = max_value_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
            -1, mppi_actions.size(1), -1, mppi_actions.size(3)
        )
        max_mppi_actions = torch.gather(mppi_actions, 2, expanded_idx).squeeze(2)
        return max_mppi_actions

    # -- Target computation ----------------------------------------------------

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[:-1], value[:-1], discount[:-1],
            bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _update_slow_target(self):
        if self._config.critic.get("slow_target", True):
            if self._updates % self._config.critic.get("slow_target_update", 1) == 0:
                mix = self._config.critic.get("slow_target_fraction", 0.02)
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
