"""WGAN discriminator with gradient penalty for inverse RL reward.

Extracted from SAILOR's WorldModel. Provides a learned reward signal that
distinguishes expert trajectories from learner trajectories.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def gradient_penalty(learner_sa, expert_sa, f, device="cuda"):
    batch_size = expert_sa.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(expert_sa)
    interpolated = Variable(
        alpha * expert_sa.data + (1 - alpha) * learner_sa.data,
        requires_grad=True,
    ).to(device)
    f_interpolated = f(interpolated.float()).mode().to(device)
    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(f_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    return ((gradients_norm - 0.4) ** 2).mean()


class Discriminator(nn.Module):
    """Moment-matching discriminator for inverse RL.

    Uses SAILOR's ``MLP`` from ``sailor.dreamer.networks`` for the
    network architecture and distribution output.
    """

    def __init__(self, feat_size, config):
        super().__init__()
        from sailor.dreamer import networks, tools

        self._config = config
        self._discrim_state_only = config.train_dp_mppi_params["discrim_state_only"]
        net_size = feat_size if self._discrim_state_only else feat_size + config.num_actions
        use_amp = config.precision == 16

        self.net = networks.MLP(
            net_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward_Discrim",
        )
        self.opt = tools.Optimizer(
            "discrim",
            self.net.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=use_amp,
        )

    def forward(self, x):
        return self.net(x)

    def get_reward(self, feat, action=None):
        if self._discrim_state_only or action is None:
            return self.net(feat)
        return self.net(torch.cat([feat, action], dim=-1))

    def update(self, post_feat, data, device):
        """Run one discriminator update step.

        Args:
            post_feat: (B, T, feat_dim) detached posterior features.
            data: preprocessed data dict with "reward" and optionally "action".
            device: torch device string.

        Returns:
            dict of scalar metrics.
        """
        metrics = {}
        expert_ids = torch.where(torch.all(data["reward"] == 1, dim=1))[0]
        learner_ids = torch.where(torch.all(data["reward"] == -1, dim=1))[0]
        assert expert_ids.shape[0] + learner_ids.shape[0] == data["reward"].shape[0]

        learner_s = post_feat[learner_ids]
        expert_s = post_feat[expert_ids]

        if self._discrim_state_only:
            learner_sa, expert_sa = learner_s, expert_s
        else:
            learner_sa = torch.cat([learner_s, data["action"][learner_ids]], -1)
            expert_sa = torch.cat([expert_s, data["action"][expert_ids]], -1)

        learner_sa = learner_sa.reshape(-1, learner_sa.shape[-1])
        expert_sa = expert_sa.reshape(-1, expert_sa.shape[-1])

        f_learner = self.net(learner_sa.float())
        f_expert = self.net(expert_sa.float())
        gp = gradient_penalty(learner_sa, expert_sa, self.net, device=device)
        pure_loss = torch.mean(f_learner.mode() - f_expert.mode())
        loss = pure_loss + 10 * gp

        metrics["discrim_gp"] = gp.item()
        metrics["discrim_pure_loss"] = pure_loss.item()
        metrics.update(self.opt(loss, self.net.parameters()))
        return metrics
