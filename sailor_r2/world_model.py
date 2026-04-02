"""R2WorldModel: r2dreamer encoder/RSSM/heads integrated into SAILOR's training interface.

Replaces SAILOR's ``WorldModel`` with r2dreamer's components while keeping
the same public API (``_train``, ``preprocess``, ``get_reward``, etc.) so
that ``Dreamer``, ``ImagBehavior``, and ``SAILORTrainer`` work unmodified.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from r2dreamer import RSSM, MLPHead, MultiDecoder, MultiEncoder, Projector
from r2dreamer import LaProp, clip_grad_agc_

from .adapter import RSSMAdapter
from .discriminator import Discriminator


def _to_np(x):
    return x.detach().cpu().numpy()


class R2WorldModel(nn.Module):
    """World model built from r2dreamer components.

    Supports two representation losses:
    - ``"r2dreamer"``: Barlow Twins (no decoder needed)
    - ``"dreamer"``: standard reconstruction through a decoder
    """

    def __init__(self, obs_space, step, config):
        super().__init__()
        self._step = step
        self._config = config
        self._use_amp = config.precision == 16
        self.rep_loss = getattr(config, "rep_loss", "r2dreamer")
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # -- Build r2dreamer model config from SAILOR-style flat config --------
        model_cfg = self._build_model_config(config, shapes)

        # -- Encoder -----------------------------------------------------------
        self.encoder = MultiEncoder(model_cfg.encoder, shapes)
        self.embed_size = self.encoder.out_dim

        # -- RSSM (wrapped in adapter) ----------------------------------------
        rssm_raw = RSSM(model_cfg.rssm, self.embed_size, config.num_actions)
        self.dynamics = RSSMAdapter(rssm_raw)
        feat_size = self.dynamics.feat_size

        # -- Heads -------------------------------------------------------------
        self.heads = nn.ModuleDict()

        # Continue head (always present)
        self.heads["cont"] = MLPHead(model_cfg.cont, feat_size)

        # Decoder (only for dreamer mode)
        if self.rep_loss == "dreamer":
            self.heads["decoder"] = MultiDecoder(
                model_cfg.decoder,
                self.dynamics._deter,
                self.dynamics.flat_stoch,
                shapes,
            )
            self._grad_heads = ["decoder", "cont"]
        else:
            self._grad_heads = ["cont"]

        # Projector (for r2dreamer / infonce modes)
        if self.rep_loss in ("r2dreamer", "infonce"):
            self.prj = Projector(feat_size, self.embed_size)
            self.barlow_lambd = float(getattr(config, "barlow_lambd", 5e-4))

        # -- Reward discriminator (SAILOR-specific) ----------------------------
        self._use_discrim = config.train_dp_mppi_params["use_discrim"]
        self.discriminator = Discriminator(feat_size, config)

        # -- Optimiser (LaProp + AGC from r2dreamer) ---------------------------
        wm_params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.heads.parameters())
        )
        if hasattr(self, "prj"):
            wm_params += list(self.prj.parameters())

        self._opt = LaProp(
            wm_params,
            lr=float(getattr(config, "model_lr", 1e-4)),
            betas=(0.9, 0.999),
            eps=1e-20,
        )
        self._agc = float(getattr(config, "agc", 0.3))
        self._agc_pmin = float(getattr(config, "agc_pmin", 1e-3))
        self._scaler = torch.amp.GradScaler(enabled=self._use_amp)

        # Loss scales
        self._loss_scales = {
            "dyn": float(config.dyn_scale),
            "rep": float(config.rep_scale),
            "cont": float(config.cont_head.get("loss_scale", 1.0)) if isinstance(config.cont_head, dict) else 1.0,
            "barlow": 0.05,
        }

        param_count = sum(p.numel() for p in wm_params)
        print(f"R2WorldModel: {param_count:,} parameters (rep_loss={self.rep_loss})")

    # -- Config bridge ---------------------------------------------------------

    @staticmethod
    def _build_model_config(config, shapes):
        """Translate SAILOR's flat config into r2dreamer's nested OmegaConf."""
        # Detect whether we have images
        has_images = any("image" in k and len(v) == 3 for k, v in shapes.items())
        has_state = any("state" in k and len(v) in (1, 2) for k, v in shapes.items())

        cnn_keys = ".*image.*" if has_images else "$^"
        mlp_keys = ".*state.*" if has_state else "$^"

        units = int(getattr(config, "units", 512))
        act = str(getattr(config, "act", "SiLU"))
        norm = True
        device = str(config.device)
        stoch = int(config.dyn_stoch)
        deter = int(config.dyn_deter)
        hidden = int(config.dyn_hidden)
        discrete = int(config.dyn_discrete)
        depth = int(getattr(config, "cnn_depth", 32))

        cfg = OmegaConf.create({
            "rssm": {
                "stoch": stoch,
                "deter": deter,
                "hidden": hidden,
                "discrete": discrete,
                "img_layers": 2,
                "obs_layers": 1,
                "dyn_layers": 1,
                "blocks": int(getattr(config, "rssm_blocks", 8)),
                "act": act,
                "unimix_ratio": float(getattr(config, "unimix_ratio", 0.01)),
                "initial": str(getattr(config, "initial", "learned")),
                "device": device,
            },
            "encoder": {
                "cnn_keys": cnn_keys,
                "mlp_keys": mlp_keys,
                "mlp": {
                    "shape": None,
                    "layers": 3,
                    "units": units,
                    "act": act,
                    "norm": norm,
                    "device": device,
                    "outscale": None,
                    "symlog_inputs": True,
                    "name": "mlp_encoder",
                },
                "cnn": {
                    "act": act,
                    "norm": norm,
                    "kernel_size": 5,
                    "minres": 4,
                    "depth": depth,
                    "mults": [2, 3, 4, 4],
                },
            },
            "decoder": {
                "cnn_keys": cnn_keys,
                "mlp_keys": mlp_keys,
                "mlp_dist": {"name": "symlog_mse"},
                "cnn_dist": {"name": "mse"},
                "mlp": {
                    "shape": None,
                    "layers": 3,
                    "units": units,
                    "act": act,
                    "norm": norm,
                    "dist": {"name": "identity"},
                    "device": device,
                    "outscale": 1.0,
                    "symlog_inputs": False,
                    "name": "mlp_decoder",
                },
                "cnn": {
                    "depth": depth,
                    "units": units,
                    "bspace": 8,
                    "mults": [2, 3, 4, 4],
                    "act": act,
                    "norm": norm,
                    "kernel_size": 5,
                    "minres": 4,
                    "outscale": 1.0,
                },
            },
            "cont": {
                "shape": [1],
                "layers": int(config.cont_head["layers"]) if isinstance(config.cont_head, dict) else 2,
                "units": units,
                "act": act,
                "norm": norm,
                "dist": {"name": "binary"},
                "outscale": 1.0,
                "device": device,
                "symlog_inputs": False,
                "name": "cont",
            },
        })
        return cfg

    # -- Parameters (exclude discriminator) ------------------------------------

    def wm_parameters(self):
        params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.heads.parameters())
        )
        if hasattr(self, "prj"):
            params += list(self.prj.parameters())
        return params

    # -- Training step ---------------------------------------------------------

    def _train(self, data):
        data = self.preprocess(data)
        metrics = {}

        with torch.amp.autocast("cuda", enabled=self._use_amp):
            embed = self.encoder(data)
            B, T = embed.shape[:2]

            post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])

            # KL loss
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior,
                free=self._config.kl_free,
                dyn_scale=self._loss_scales["dyn"],
                rep_scale=self._loss_scales["rep"],
            )
            assert kl_loss.shape == (B, T), kl_loss.shape

            feat = self.dynamics.get_feat(post)
            losses = {}

            # Head losses (decoder if dreamer mode, cont always)
            for name, head in self.heads.items():
                grad_head = name in self._grad_heads
                inp = feat if grad_head else feat.detach()
                pred = head(inp)
                if isinstance(pred, dict):
                    for k, dist in pred.items():
                        losses[k] = -dist.log_prob(data[k])
                else:
                    losses[name] = -pred.log_prob(data[name])

            # Representation loss
            if self.rep_loss == "r2dreamer":
                x1 = self.prj(feat.reshape(B * T, -1))
                x2 = embed.reshape(B * T, -1).detach()
                x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
                x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
                c = torch.mm(x1_norm.T, x2_norm) / (B * T)
                invariance = (torch.diagonal(c) - 1.0).pow(2).sum()
                off_diag = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
                redundancy = c[off_diag].pow(2).sum()
                losses["barlow"] = (invariance + self.barlow_lambd * redundancy).expand(B, T)
            elif self.rep_loss == "infonce":
                x1 = self.prj(feat.reshape(B * T, -1))
                x2 = embed.reshape(B * T, -1).detach()
                logits = torch.matmul(x1, x2.T)
                logits = logits - logits.max(dim=1, keepdim=True).values
                labels = torch.arange(B * T, device=x1.device)
                nce = torch.nn.functional.cross_entropy(logits, labels)
                losses["infonce"] = nce.expand(B, T)

            # Combine losses
            scaled = {
                k: v * self._loss_scales.get(k, 1.0)
                for k, v in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            total_loss = torch.mean(model_loss)

        # Backward + AGC + step
        self._opt.zero_grad()
        self._scaler.scale(total_loss).backward()
        self._scaler.unscale_(self._opt)
        clip_grad_agc_(self.wm_parameters(), self._agc, self._agc_pmin)
        self._scaler.step(self._opt)
        self._scaler.update()

        # Metrics
        metrics["model_loss"] = total_loss.item()
        for name, loss in losses.items():
            metrics[f"{name}_loss"] = _to_np(torch.mean(loss))
        metrics["kl"] = _to_np(torch.mean(kl_value))
        metrics["dyn_loss"] = _to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = _to_np(torch.mean(rep_loss))
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            metrics["prior_ent"] = _to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics["post_ent"] = _to_np(torch.mean(self.dynamics.get_dist(post).entropy()))

        post_detached = {k: v.detach() for k, v in post.items()}

        # Discriminator update
        self._step += 1
        discrim_every = self._config.train_dp_mppi_params.get("upate_discrim_every", 100)
        if self._use_discrim and self._step % discrim_every == 0:
            feat_detached = self.dynamics.get_feat(post_detached)
            metrics.update(
                self.discriminator.update(feat_detached, data, self._config.device)
            )

        context = {
            "embed": embed,
            "feat": self.dynamics.get_feat(post_detached),
            "kl": kl_value,
            "postent": self.dynamics.get_dist(post).entropy(),
        }
        return post_detached, context, metrics

    def get_reward(self, data):
        if self._use_discrim:
            return self.discriminator.get_reward(data)
        return self.heads.get("reward", self.discriminator.net)(data)

    # -- Preprocessing (SAILOR-compatible) -------------------------------------

    def preprocess(self, obs):
        obs = obs.copy()
        # Remove stacking dimension
        if len(obs["state"].shape) == 4:
            for key in obs:
                if "image" in key:
                    obs[key] = obs[key][..., -1]
            obs["state"] = obs["state"][..., -1]
        if "action" in obs and len(obs["action"].shape) == 4:
            obs["action"] = obs["action"][..., 0]

        for key in obs:
            if "image" in key:
                obs[key] = torch.Tensor(obs[key]) / 255.0
        if "discount" in obs:
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1) * self._config.discount
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs
