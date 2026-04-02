"""SAILORTrainer adapted for sailor-r2.

This is a thin subclass of SAILOR's ``SAILORTrainer`` that swaps in our
``Dreamer`` (which uses ``R2WorldModel``) instead of SAILOR's original.
All training logic (warmstart, collect, train_wm_critic, relabel, etc.)
is inherited unchanged.
"""

from sailor.sailor_trainer import SAILORTrainer as _BaseSAILORTrainer
from sailor.dreamer import tools

from .dreamer_class import Dreamer


class SAILORTrainer(_BaseSAILORTrainer):
    """SAILOR trainer using R2-Dreamer world model."""

    def __init__(self, *args, **kwargs):
        # Temporarily prevent base __init__ from creating the dreamer_class
        # so we can replace it with our version.
        super().__init__(*args, **kwargs)

        # Replace SAILOR's Dreamer with our R2-Dreamer version
        del self.dreamer_class
        import torch
        torch.cuda.empty_cache()

        self.dreamer_class = Dreamer(
            obs_space=self.eval_envs.observation_space,
            base_policy=self.base_policy,
            config=self.config,
            logger=None,
            dataset=None,
            expert_dataset=self.expert_datset,
        ).to(self.config.device)

        # Re-setup oracle reliability if configured
        ur_cfg = getattr(self.config, "unreliable_region", None)
        if ur_cfg and ur_cfg.get("oracle_avoidance", False):
            from sailor.dreamer.reliability import OracleReliabilityFn
            from environments.maniskill.unreliable_region_wrapper import EE_STATE_START_IDX

            _, task = self.config.task.split("__", 1)
            reliability_fn = OracleReliabilityFn(
                bad_region_axis=ur_cfg.get("axis", 1),
                bad_region_threshold=ur_cfg.get("threshold", 0.0),
                bad_region_side=ur_cfg.get("side", "positive"),
                ee_state_start_idx=EE_STATE_START_IDX.get(task.lower(), 9),
            )
            tb = self.dreamer_class._task_behavior
            if hasattr(tb, "_orig_mod"):
                tb._orig_mod.reliability_fn = reliability_fn
            else:
                tb.reliability_fn = reliability_fn

        # Re-create ResidualPolicy with the new dreamer_class
        from sailor.policies.residual_policy import ResidualPolicy
        self.residual_policy = ResidualPolicy(
            config=self.config,
            dreamer_class=self.dreamer_class,
            expert_eps=self.expert_eps,
            train_eps=self.train_eps,
            train_env=self.train_env,
            eval_envs=self.eval_envs,
            logger=self.logger,
        )
