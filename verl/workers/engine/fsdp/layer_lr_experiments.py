# Copyright 2026 Zhijie Xia, Qihong (OPD foresight project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Per-layer / per-block learning-rate experiments for the FSDP engine.

Motivated by "Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy
Distillation" (Cai et al., 2026): reasoning-relevant updates concentrate in middle-layer
MLPs (Property 1) and dominant update directions lock in early (Property 2).

Everything here is driven by ``FSDPOptimizerConfig`` flags.  When no flag is set the
controller is inert and the engine behaves exactly like upstream verl.

Experiments
-----------
fixed profiles (``layerwise_lr_enabled``)
    A depth profile of LR multipliers: hard window (the original "layerwise" run), Gaussian
    or raised-cosine bump; separate multipliers for the middle, the periphery and the
    non-layer parameters; restricted to all / MLP / attention modules.  Multiplier 0 freezes.

adaptive per-layer multipliers (``adaptive_layer_lr_enabled``)
    score = "topk_energy"          top-k singular energy of the per-step gradient (original run)
    score = "direction_stability"  cosine between successive per-layer weight displacements
                                   (Property 2 at layer level; no SVD, fp32 CPU snapshots)
    mapping = "minmax"             1 + minmax(score) * (max - 1)        (original run)
    mapping = "prior_exp"          prior * exp(gamma * z), clamped, renormalised to a fixed
                                   mean multiplier (budget-neutral reallocation)
    fit_profile                    replaces the raw per-layer z by a fitted Gaussian bump
                                   (3 parameters instead of L free multipliers)

utility probes (``utility_probe_enabled``)
    Every K optimizer updates, on the current mini-batch, evaluate a policy-gradient surrogate
    J(W) = sum_t A_t log pi_W(y_t) / sum_t |A_t| with block b's accumulated update removed
    (leave-one-block-out) or injected alone into the base weights.  u_b = marginal utility of
    block b (optionally per unit ||dW_b||) -> per-block multipliers via the same prior_exp map.

block-wise extrapolation (``block_extrap_enabled``)
    EffOPD-style: at optimizer updates t = 2^n (or every K), for each block propose
    W_b + alpha * (W_b - W_b,prev) for alpha in an increasing list and keep the largest alpha
    whose probe J does not drop (and whose k3-KL to the pre-extrapolation policy stays below a
    bound).  Optionally feeds accepted alphas back into the LR multipliers.

Single-GPU (fsdp_size=1, use_orig_params=True) is required for the snapshot-based features,
exactly as for the original Top-k experiments in this branch.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def param_kind(name: str) -> str:
    """Classify a parameter name into 'mlp', 'attn' or 'other' (norms, embeddings, lm_head)."""
    if ".mlp." in name:
        return "mlp"
    if "self_attn" in name or ".attention." in name:
        return "attn"
    return "other"


def in_scope(kind: str, scope: str) -> bool:
    """scope in {'all', 'mlp', 'attn'}."""
    if scope == "all":
        return True
    return kind == scope


def parse_float_list(value) -> list[float]:
    """'2,4,8' / [2, 4, 8] / (2, 4, 8) -> [2.0, 4.0, 8.0]."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        return [float(p) for p in parts if p]
    return [float(v) for v in value]


def block_bounds(num_layers: int, num_blocks: int) -> list[tuple[int, int]]:
    """Split [0, L) into num_blocks contiguous, near-equal [start, end) ranges."""
    num_blocks = max(1, min(num_blocks, num_layers))
    edges = [round(i * num_layers / num_blocks) for i in range(num_blocks + 1)]
    return [(edges[i], edges[i + 1]) for i in range(num_blocks)]


def layer_to_block(num_layers: int, bounds: list[tuple[int, int]]) -> list[int]:
    out = [0] * num_layers
    for b, (s, e) in enumerate(bounds):
        for l in range(s, e):
            out[l] = b
    return out


def depth_bump(num_layers: int, profile: str, start_frac: float, end_frac: float, center_frac: float,
               width_frac: float) -> torch.Tensor:
    """Return bump(l) in [0, 1] for l = 0..L-1.

    window:   1 for floor(L*start_frac) <= l < ceil(L*end_frac), else 0  (original layerwise run)
    gaussian: exp(-0.5 * ((l - c) / w)^2),        c = center_frac*(L-1), w = width_frac*L
    cosine:   0.5 * (1 + cos(pi * (l - c) / w)) for |l - c| < w, else 0
    """
    l = torch.arange(num_layers, dtype=torch.float64)
    if profile == "window":
        s = math.floor(num_layers * start_frac)
        e = math.ceil(num_layers * end_frac)
        return ((l >= s) & (l < e)).to(torch.float64)
    c = center_frac * (num_layers - 1)
    w = max(width_frac * num_layers, 1e-6)
    x = (l - c) / w
    if profile == "gaussian":
        return torch.exp(-0.5 * x * x)
    if profile == "cosine":
        return torch.where(x.abs() < 1.0, 0.5 * (1.0 + torch.cos(math.pi * x)), torch.zeros_like(x))
    raise ValueError(f"unknown layerwise_profile={profile!r} (window | gaussian | cosine)")


def standardize(x: torch.Tensor, clamp: float = 2.0) -> torch.Tensor:
    z = (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
    return z.clamp(-clamp, clamp)


def fit_gaussian_bump(z: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Least-squares fit of z(l) ~ a + h * exp(-0.5((l-c)/w)^2), h >= 0, by a small grid search.

    Returns the smoothed curve and the fitted parameters.  Used to replace L noisy per-layer
    values by a 3-parameter inverted-U (Section 5.4 of the notes)."""
    L = z.numel()
    l = torch.arange(L, dtype=torch.float64)
    zz = z.to(torch.float64)
    best = None
    centers = torch.arange(0.0, L - 1 + 1e-9, 0.5, dtype=torch.float64)
    widths = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    for c in centers.tolist():
        for w in widths:
            g = torch.exp(-0.5 * ((l - c) / w) ** 2)
            # solve [1 g] [a h]^T = z in least squares
            X = torch.stack([torch.ones_like(g), g], dim=1)
            sol = torch.linalg.lstsq(X, zz.unsqueeze(1)).solution.squeeze(1)
            a, h = sol[0].item(), sol[1].item()
            if h < 0:
                continue
            resid = ((X @ sol) - zz).pow(2).sum().item()
            if best is None or resid < best[0]:
                best = (resid, a, h, c, w)
    if best is None:  # z is (numerically) a valley everywhere: fall back to a flat curve
        return torch.full_like(zz, zz.mean().item()).to(z.dtype), {"a": zz.mean().item(), "h": 0.0, "c": 0.0, "w": 0.0}
    _, a, h, c, w = best
    curve = a + h * torch.exp(-0.5 * ((l - c) / w) ** 2)
    return curve.to(z.dtype), {"a": a, "h": h, "c": c, "w": w}


def _k3_kl(logp_ref: torch.Tensor, logp_new: torch.Tensor, mask: torch.Tensor) -> float:
    """k3 estimator of KL(pi_ref || pi_new) on samples from pi_ref (verl's low_var_kl)."""
    r = (logp_new - logp_ref).clamp(-20.0, 20.0)
    k3 = torch.exp(r) - 1.0 - r
    return (k3 * mask).sum().item() / max(mask.sum().item(), 1.0)


# ----------------------------------------------------------------------------------------------
# controller
# ----------------------------------------------------------------------------------------------
class LayerLRController:
    """Owns param groups, LR multipliers, probes and extrapolation for one FSDP engine."""

    def __init__(self, engine, module: torch.nn.Module):
        self.engine = engine
        self.cfg = engine.optimizer_config
        self.rank = getattr(engine, "rank", 0)
        cfg = self.cfg

        self.fixed_enabled = bool(getattr(cfg, "layerwise_lr_enabled", False))
        self.adaptive_enabled = bool(getattr(cfg, "adaptive_layer_lr_enabled", False))
        self.utility_enabled = bool(getattr(cfg, "utility_probe_enabled", False))
        self.extrap_enabled = bool(getattr(cfg, "block_extrap_enabled", False))
        self.enabled = self.fixed_enabled or self.adaptive_enabled or self.utility_enabled or self.extrap_enabled

        self.step = 0  # optimizer updates completed
        self._metrics: dict[str, float] = {}
        self._module = module

        if not self.enabled:
            return

        self.L = int(engine.model_config.hf_config.num_hidden_layers)

        needs_snapshots = self.utility_enabled or self.extrap_enabled or (
            self.adaptive_enabled and cfg.adaptive_layer_lr_score == "direction_stability"
        )
        if needs_snapshots and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError(
                "[LayerLR] snapshot-based experiments (direction_stability / utility_probe / block_extrap) "
                "need the full parameter tensors: run with a single GPU, fsdp_size=1 and use_orig_params=true."
            )

        # ---------------------------------------------------------------- parameter bookkeeping
        # (name, layer_idx, kind, ndim) for every trainable parameter inside a transformer layer.
        # Parameters are re-resolved by name from the live module at every event (FSDP may
        # re-register parameter objects), see _live_params().
        self._layer_entries: list[tuple[str, int, str, int]] = []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            m = _LAYER_RE.search(name)
            if m is None:
                continue
            l = int(m.group(1))
            if l >= self.L:
                raise RuntimeError(f"[LayerLR] layer index {l} in {name} >= num_hidden_layers={self.L}")
            self._layer_entries.append((name, l, param_kind(name), p.ndim))
        if not self._layer_entries:
            raise RuntimeError("[LayerLR] no parameters matched 'layers.<i>.' -- check parameter names.")

        # ---------------------------------------------------------------- fixed profile
        self.bump = depth_bump(
            self.L,
            getattr(cfg, "layerwise_profile", "window"),
            cfg.middle_layer_start_frac,
            cfg.middle_layer_end_frac,
            getattr(cfg, "layerwise_center_frac", 0.5),
            getattr(cfg, "layerwise_width_frac", 1.0 / 6.0),
        )
        self.periphery_mult = float(getattr(cfg, "periphery_lr_multiplier", 1.0))
        self.other_mult = float(getattr(cfg, "other_lr_multiplier", 1.0))
        self.fixed_scope = getattr(cfg, "layerwise_module_scope", "all")
        # per-layer multiplier of the fixed profile (for kinds inside the fixed scope)
        self.fixed_layer_mult = self.periphery_mult + (float(cfg.middle_lr_multiplier) - self.periphery_mult) * self.bump

        # ---------------------------------------------------------------- adaptive
        self.score_name = getattr(cfg, "adaptive_layer_lr_score", "topk_energy")
        self.score_scope = getattr(cfg, "adaptive_layer_lr_score_scope", "all")
        self.apply_scope = getattr(cfg, "adaptive_layer_lr_apply_scope", "all")
        self.mapping = getattr(cfg, "adaptive_layer_lr_mapping", "minmax")
        self.prior_name = getattr(cfg, "adaptive_layer_lr_prior", "none")
        if self.fixed_enabled and self.prior_name == "none" and (self.adaptive_enabled or self.utility_enabled):
            # layerwise on + adaptive on == adapt around the layerwise profile
            self.prior_name = "profile"
        self.gamma = float(getattr(cfg, "adaptive_layer_lr_gamma", 0.25))
        self.target_mean = float(getattr(cfg, "adaptive_layer_lr_target_mean", -1.0))
        self.min_mult = float(getattr(cfg, "adaptive_layer_lr_min_multiplier", 0.5))
        self.max_mult = float(cfg.adaptive_layer_lr_max_multiplier)
        self.fit_profile = bool(getattr(cfg, "adaptive_layer_lr_fit_profile", False))
        self.stability_ref = getattr(cfg, "adaptive_layer_lr_stability_ref", "prev_window")
        self.ema_beta = float(cfg.adaptive_layer_lr_ema_beta)
        self.interval = int(cfg.adaptive_layer_lr_interval)

        self._score_ema: Optional[torch.Tensor] = None  # [L] EMA of the raw score
        self._stab_z: Optional[torch.Tensor] = None  # [L] standardized stability signal
        self._w_prev: Optional[dict[str, torch.Tensor]] = None  # fp32 CPU snapshot (scored params)
        self._d_prev: dict[str, torch.Tensor] = {}  # bf16 CPU previous displacement
        self._w0_scored: Optional[dict[str, torch.Tensor]] = None  # for stability_ref == cumulative
        self.layer_mult: Optional[torch.Tensor] = None  # [L] current adaptive multipliers (None -> prior)

        self._scored_entries = [
            e for e in self._layer_entries if e[3] == 2 and e[2] in ("mlp", "attn") and in_scope(e[2], self.score_scope)
        ]

        # ---------------------------------------------------------------- utility probe
        self.probe_objective = getattr(cfg, "probe_objective", "adv_logp")
        self.util_interval = int(getattr(cfg, "utility_probe_interval", 20))
        self.util_num_blocks = int(getattr(cfg, "utility_probe_num_blocks", 4))
        self.util_scope = getattr(cfg, "utility_probe_module_scope", "all")
        self.util_mode = getattr(cfg, "utility_probe_mode", "leave_one_out")
        self.util_ema_beta = float(getattr(cfg, "utility_probe_ema_beta", 0.7))
        self.util_norm = bool(getattr(cfg, "utility_probe_normalize_by_norm", True))
        self.util_gamma = float(getattr(cfg, "utility_probe_gamma", 0.5))
        self.util_bounds = block_bounds(self.L, self.util_num_blocks)
        self._util_ema: Optional[torch.Tensor] = None  # [B]
        self._util_z: Optional[torch.Tensor] = None  # [L]
        self._last_util_step = -1

        # ---------------------------------------------------------------- block extrapolation
        self.ex_schedule = getattr(cfg, "block_extrap_schedule", "exp")
        self.ex_interval = int(getattr(cfg, "block_extrap_interval", 50))
        self.ex_num_blocks = int(getattr(cfg, "block_extrap_num_blocks", 4))
        self.ex_scope = getattr(cfg, "block_extrap_module_scope", "all")
        self.ex_alphas = sorted(parse_float_list(getattr(cfg, "block_extrap_alphas", "2,4,8")))
        self.ex_tol = float(getattr(cfg, "block_extrap_tolerance", 0.0))
        self.ex_max_kl = float(getattr(cfg, "block_extrap_max_kl", 0.05))
        self.ex_order = getattr(cfg, "block_extrap_order", "middle_out")
        self.ex_feedback = bool(getattr(cfg, "block_extrap_feedback_lr", False))
        self.ex_first_step = int(getattr(cfg, "block_extrap_first_step", 1))
        self.ex_max_step = int(getattr(cfg, "block_extrap_max_step", -1))
        self.ex_bounds = block_bounds(self.L, self.ex_num_blocks)
        self._next_extrap_step = self.ex_first_step  # exp schedule
        self._last_extrap_step = -1  # interval schedule
        self._w_ckpt: Optional[dict[str, torch.Tensor]] = None  # fp32 CPU snapshot at previous checkpoint
        self._ex_mult: Optional[torch.Tensor] = None  # [L] multipliers fed back from accepted alphas

        # ---------------------------------------------------------------- snapshots
        self._w0: Optional[dict[str, torch.Tensor]] = None
        if self.utility_enabled:
            self._w0 = self._snapshot(self._block_entries(self.util_scope))
        if self.extrap_enabled:
            self._w_ckpt = self._snapshot(self._block_entries(self.ex_scope))
        if self.adaptive_enabled and self.score_name == "direction_stability" and self.stability_ref == "cumulative":
            self._w0_scored = self._snapshot(self._scored_entries)

        if self.rank == 0:
            print(
                f"[LayerLR] L={self.L} fixed={self.fixed_enabled} adaptive={self.adaptive_enabled}"
                f"(score={self.score_name}, mapping={self.mapping}, prior={self.prior_name}) "
                f"utility={self.utility_enabled}(blocks={self.util_num_blocks}, mode={self.util_mode}) "
                f"extrap={self.extrap_enabled}(schedule={self.ex_schedule}, alphas={self.ex_alphas}, blocks={self.ex_num_blocks})"
            )
            print(f"[LayerLR] fixed per-layer multipliers: {[round(x, 3) for x in self.fixed_layer_mult.tolist()]}")

    # ------------------------------------------------------------------------------------------
    # parameter groups
    # ------------------------------------------------------------------------------------------
    def build_param_groups(self, module: torch.nn.Module) -> list[dict]:
        """One optimizer group per (layer, kind) plus one for the non-layer parameters."""
        base_lr = float(self.cfg.lr)
        buckets: dict[tuple[int, str], list[torch.nn.Parameter]] = {}
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            m = _LAYER_RE.search(name)
            key = (-1, "other") if m is None else (int(m.group(1)), param_kind(name))
            buckets.setdefault(key, []).append(p)
        groups = []
        for (l, kind) in sorted(buckets.keys()):
            groups.append({"params": buckets[(l, kind)], "lr": base_lr, "layer_idx": l, "kind": kind})
        if self.rank == 0:
            print(f"[LayerLR] built {len(groups)} param groups (base_lr={base_lr:.3e})")
        return groups

    # ------------------------------------------------------------------------------------------
    # multipliers
    # ------------------------------------------------------------------------------------------
    def _prior(self) -> torch.Tensor:
        if self.prior_name == "profile":
            return self.fixed_layer_mult.clone()
        return torch.ones(self.L, dtype=torch.float64)

    def _map_signal_to_multipliers(self, z: torch.Tensor, gamma: float) -> torch.Tensor:
        """prior * exp(gamma * z), clamped to [min, max], renormalised to target mean (if > 0)."""
        m = self._prior() * torch.exp(gamma * z.to(torch.float64))
        m = m.clamp(self.min_mult, self.max_mult)
        if self.target_mean > 0:
            m = m * (self.target_mean * self.L / m.sum())
            m = m.clamp(self.min_mult, self.max_mult)
        return m

    def _recompute_layer_mult(self):
        """Combine the available signals into self.layer_mult ([L])."""
        if self._ex_mult is not None and self.ex_feedback:
            self.layer_mult = self._ex_mult.clone()
            return
        if self.adaptive_enabled and self.mapping == "minmax":
            if self._score_ema is None:
                self.layer_mult = None
                return
            s = self._score_ema.to(torch.float64)
            normalized = (s - s.min()) / (s.max() - s.min() + 1e-12)
            self.layer_mult = 1.0 + normalized * (self.max_mult - 1.0)
            return
        # signal s_l = gamma * z_stability + util_gamma * z_utility  (each z standardized over layers)
        s = torch.zeros(self.L, dtype=torch.float64)
        have = False
        if self._stab_z is not None:
            s = s + self.gamma * self._stab_z.to(torch.float64)
            have = True
        if self._util_z is not None:
            s = s + self.util_gamma * self._util_z.to(torch.float64)
            have = True
        if not have:
            self.layer_mult = None
            return
        if self.fit_profile:
            s, params = fit_gaussian_bump(s)
            self._metrics["layer_lr/fit/center"] = float(params["c"])
            self._metrics["layer_lr/fit/width"] = float(params["w"])
            self._metrics["layer_lr/fit/height"] = float(params["h"])
        self.layer_mult = self._map_signal_to_multipliers(s, 1.0)

    def group_multiplier(self, layer_idx: int, kind: str) -> float:
        """Total LR multiplier for one param group."""
        if layer_idx < 0:
            return self.other_mult if self.fixed_enabled else 1.0
        adaptive_active = self.adaptive_enabled or self.utility_enabled or (self.extrap_enabled and self.ex_feedback)
        if adaptive_active:
            base = self.fixed_layer_mult[layer_idx].item() if (self.fixed_enabled and in_scope(kind, self.fixed_scope)) else 1.0
            if self.layer_mult is None:
                # before the first measurement: the prior
                if self.prior_name == "profile" and in_scope(kind, self.apply_scope):
                    return self.fixed_layer_mult[layer_idx].item()
                return base
            if in_scope(kind, self.apply_scope):
                return self.layer_mult[layer_idx].item()
            return base
        if self.fixed_enabled:
            return self.fixed_layer_mult[layer_idx].item() if in_scope(kind, self.fixed_scope) else 1.0
        return 1.0

    @torch.no_grad()
    def apply_lr(self):
        """Set group['lr'] = scheduler lr * multiplier for every group (scheduler owns the schedule)."""
        if not self.enabled:
            return
        opt = self.engine.optimizer
        sched = getattr(self.engine, "lr_scheduler", None)
        if sched is not None:
            scheduled = sched.get_last_lr()
            if len(scheduled) != len(opt.param_groups):
                raise RuntimeError("[LayerLR] scheduler/optimizer parameter-group count mismatch.")
        else:
            scheduled = [float(self.cfg.lr)] * len(opt.param_groups)
        for g, lr in zip(opt.param_groups, scheduled):
            g["lr"] = lr * self.group_multiplier(g.get("layer_idx", -1), g.get("kind", "other"))
        # expose the effective per-layer multiplier (max over kinds) for logging
        for l in range(self.L):
            self._metrics[f"layer_lr/mult/l{l:02d}"] = max(self.group_multiplier(l, k) for k in ("mlp", "attn", "other"))

    # ------------------------------------------------------------------------------------------
    # engine hooks
    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def on_optimizer_step(self):
        """Called in optimizer_step() before gradient clipping (gradients are available)."""
        if not self.enabled:
            return
        self.step += 1
        if self.adaptive_enabled and self.step % self.interval == 0:
            t0 = time.time()
            if self.score_name == "topk_energy":
                self._update_topk_energy()
            elif self.score_name == "direction_stability":
                self._update_direction_stability()
            else:
                raise ValueError(f"unknown adaptive_layer_lr_score={self.score_name!r}")
            self._recompute_layer_mult()
            self._metrics["layer_lr/score_time_s"] = time.time() - t0
            if self.rank == 0 and self.layer_mult is not None:
                print(f"[LayerLR] step={self.step} multipliers={[round(x, 3) for x in self.layer_mult.tolist()]}")
        self.apply_lr()

    @torch.no_grad()
    def on_lr_scheduler_step(self):
        self.apply_lr()

    @torch.no_grad()
    def on_train_batch_begin(self, data):
        """Called at the start of engine.train_batch(mini_batch) -- before zero_grad / forward."""
        if not self.enabled:
            return
        if self.utility_enabled and self.step > 0 and self.step % self.util_interval == 0 and self._last_util_step != self.step:
            self._last_util_step = self.step
            t0 = time.time()
            self._run_utility_probe(data)
            self._recompute_layer_mult()
            self.apply_lr()
            self._metrics["layer_lr/utility_time_s"] = time.time() - t0
        if self.extrap_enabled and self._extrap_due():
            t0 = time.time()
            self._run_block_extrapolation(data)
            if self.ex_feedback:
                self._recompute_layer_mult()
                self.apply_lr()
            self._metrics["layer_lr/extrap_time_s"] = time.time() - t0

    def pop_metrics(self) -> dict[str, float]:
        """Latest state (multipliers, scores, utilities, accepted alphas) as scalar metrics."""
        if not self.enabled:
            return {}
        return {k: float(v) for k, v in self._metrics.items()}

    # ------------------------------------------------------------------------------------------
    # scores
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _to_cpu(p: torch.Tensor) -> torch.Tensor:
        # always a fresh fp32 copy (``.cpu()`` / ``.float()`` alias when already CPU / fp32)
        return p.detach().to(device="cpu", dtype=torch.float32, copy=True)

    def _live_params(self) -> dict[str, torch.nn.Parameter]:
        return dict(self._module.named_parameters())

    def _snapshot(self, entries) -> dict[str, torch.Tensor]:
        live = self._live_params()
        return {name: self._to_cpu(live[name]) for name, _, _, _ in entries}

    def _update_topk_energy(self):
        """Original experiment: sqrt(sum of top-k singular values^2) of each layer's gradients."""
        from torch.distributed.tensor import DTensor

        ratio = float(self.cfg.adaptive_layer_lr_topk_ratio)
        energy = torch.zeros(self.L, dtype=torch.float64)
        live = self._live_params()
        for name, l, kind, _ in self._scored_entries:
            g = live[name].grad
            if g is None:
                continue
            if isinstance(g, DTensor):
                raise RuntimeError(f"[LayerLR] {name} gradient is a DTensor; use fsdp_size=1 and use_orig_params=true.")
            G = g.detach().float()
            if not torch.isfinite(G).all():
                continue
            S = torch.linalg.svdvals(G)
            k = max(1, math.ceil(ratio * S.numel()))
            energy[l] += torch.sum(S[:k].double() ** 2).cpu()
        scores = torch.sqrt(energy)
        self._ema_update(scores)
        for l in range(self.L):
            self._metrics[f"layer_lr/score/l{l:02d}"] = self._score_ema[l].item()

    def _update_direction_stability(self):
        """cos(D_l^(n), D_l^(n-1)) with D^(n) = W(t_n) - W(t_{n-1}) (or W(t_n) - W_0 as reference)."""
        dots = torch.zeros(self.L, dtype=torch.float64)
        n_cur = torch.zeros(self.L, dtype=torch.float64)
        n_ref = torch.zeros(self.L, dtype=torch.float64)
        new_snap: dict[str, torch.Tensor] = {}
        have_ref = False
        live = self._live_params()
        for name, l, kind, _ in self._scored_entries:
            w = self._to_cpu(live[name])
            new_snap[name] = w
            if self._w_prev is None:
                continue
            d = w - self._w_prev[name]  # displacement over the last window (fp32)
            if self.stability_ref == "cumulative":
                ref = w - self._w0_scored[name]
            else:
                ref = self._d_prev.get(name)
                ref = None if ref is None else ref.float()
            if ref is not None:
                have_ref = True
                dots[l] += torch.dot(d.flatten(), ref.flatten()).double()
                n_cur[l] += torch.dot(d.flatten(), d.flatten()).double()
                n_ref[l] += torch.dot(ref.flatten(), ref.flatten()).double()
            if self.stability_ref != "cumulative":
                self._d_prev[name] = d.to(torch.bfloat16)
        self._w_prev = new_snap
        if not have_ref:
            return  # need two displacements (or one displacement + W_0) before scoring
        cos = dots / (torch.sqrt(n_cur) * torch.sqrt(n_ref) + 1e-30)
        cos = torch.nan_to_num(cos, nan=0.0).clamp(-1.0, 1.0)
        self._ema_update(cos)
        self._stab_z = standardize(self._score_ema)
        for l in range(self.L):
            self._metrics[f"layer_lr/score/l{l:02d}"] = self._score_ema[l].item()
        if self.rank == 0:
            print(f"[LayerLR] step={self.step} stability cos={[round(x, 3) for x in self._score_ema.tolist()]}")

    def _ema_update(self, scores: torch.Tensor):
        scores = scores.to(torch.float64)
        if self._score_ema is None:
            self._score_ema = scores.clone()
        else:
            self._score_ema.mul_(self.ema_beta).add_(scores, alpha=1.0 - self.ema_beta)

    # ------------------------------------------------------------------------------------------
    # probes
    # ------------------------------------------------------------------------------------------
    def _block_entries(self, scope: str):
        return [e for e in self._layer_entries if in_scope(e[2], scope)]

    def _block_params(self, bounds: tuple[int, int], scope: str):
        """(name, live parameter) for the layers in [start, end) whose kind is in scope."""
        s, e = bounds
        live = self._live_params()
        return [(name, live[name]) for name, l, kind, _ in self._layer_entries if s <= l < e and in_scope(kind, scope)]

    @torch.no_grad()
    def _probe(self, data) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Forward-only pass on the mini-batch.  Returns (J, logp [bsz, R], mask [bsz, R]).

        J = sum_t A_t log pi(y_t) / sum_t |A_t|  ('adv_logp', the REINFORCE surrogate GRPO/PG-OPD
        optimise) or the mean log-prob of positive-advantage tokens ('pos_logp')."""
        from verl.workers.utils.padding import no_padding_2_padding

        outs = self.engine.forward_backward_batch(data, loss_function=None, forward_only=True)
        logp = no_padding_2_padding(outs["model_output"]["log_probs"], data).float()  # (bsz, R)
        sel = data.select("response_mask", "advantages").to_padded_tensor()
        mask = sel["response_mask"].to(logp.device).bool()
        adv = sel["advantages"].to(logp.device).float()
        R = min(logp.shape[1], mask.shape[1], adv.shape[1])
        logp, mask, adv = logp[:, :R], mask[:, :R], adv[:, :R]
        if self.probe_objective == "pos_logp":
            m = mask & (adv > 0)
            J = (logp * m).sum() / m.sum().clamp(min=1)
        else:
            J = (adv * logp * mask).sum() / ((adv.abs() * mask).sum() + 1e-8)
        return J.item(), logp, mask

    @torch.no_grad()
    def _run_utility_probe(self, data):
        B = len(self.util_bounds)
        util = torch.zeros(B, dtype=torch.float64)
        norms = torch.zeros(B, dtype=torch.float64)
        if self.util_mode == "leave_one_out":
            J_full, _, _ = self._probe(data)
            for b, bounds in enumerate(self.util_bounds):
                params = self._block_params(bounds, self.util_scope)
                backup = {}
                for name, p in params:
                    backup[name] = self._to_cpu(p)
                    w0 = self._w0[name]
                    norms[b] += (backup[name] - w0).pow(2).sum().double()
                    p.copy_(w0.to(p.device))
                J_wo, _, _ = self._probe(data)
                for name, p in params:
                    p.copy_(backup[name].to(p.device))
                util[b] = J_full - J_wo  # what block b's accumulated update buys
                self._metrics[f"layer_lr/probe/J_without_b{b}"] = J_wo
            self._metrics["layer_lr/probe/J_full"] = J_full
        elif self.util_mode == "inject":
            # W_0 + dW_b only (the paper's sliding-window injection)
            all_params = self._block_params((0, self.L), self.util_scope)
            backup = {name: self._to_cpu(p) for name, p in all_params}
            for name, p in all_params:
                p.copy_(self._w0[name].to(p.device))
            J_base, _, _ = self._probe(data)
            for b, bounds in enumerate(self.util_bounds):
                params = self._block_params(bounds, self.util_scope)
                for name, p in params:
                    p.copy_(backup[name].to(p.device))
                    norms[b] += (backup[name] - self._w0[name]).pow(2).sum().double()
                J_b, _, _ = self._probe(data)
                for name, p in params:
                    p.copy_(self._w0[name].to(p.device))
                util[b] = J_b - J_base
                self._metrics[f"layer_lr/probe/J_inject_b{b}"] = J_b
            for name, p in all_params:
                p.copy_(backup[name].to(p.device))
            self._metrics["layer_lr/probe/J_base"] = J_base
        else:
            raise ValueError(f"unknown utility_probe_mode={self.util_mode!r}")
        norms = norms.sqrt()
        if self.util_norm:
            util = util / (norms + 1e-12)
        if self._util_ema is None:
            self._util_ema = util.clone()
        else:
            self._util_ema.mul_(self.util_ema_beta).add_(util, alpha=1.0 - self.util_ema_beta)
        zb = standardize(self._util_ema)
        l2b = layer_to_block(self.L, self.util_bounds)
        self._util_z = torch.tensor([zb[l2b[l]].item() for l in range(self.L)], dtype=torch.float64)
        for b in range(B):
            self._metrics[f"layer_lr/utility/b{b}"] = util[b].item()
            self._metrics[f"layer_lr/utility_ema/b{b}"] = self._util_ema[b].item()
            self._metrics[f"layer_lr/dw_norm/b{b}"] = norms[b].item()
        if self.rank == 0:
            print(
                f"[LayerLR] step={self.step} utility={[f'{x:.3e}' for x in util.tolist()]} "
                f"|dW|={[f'{x:.3e}' for x in norms.tolist()]}"
            )

    # ------------------------------------------------------------------------------------------
    # block-wise extrapolation
    # ------------------------------------------------------------------------------------------
    def _extrap_due(self) -> bool:
        if self.ex_max_step > 0 and self.step > self.ex_max_step:
            return False
        if self.ex_schedule == "exp":
            return self.step == self._next_extrap_step
        if self.ex_schedule == "interval":
            return self.step > 0 and self.step % self.ex_interval == 0 and self.step != self._last_extrap_step
        raise ValueError(f"unknown block_extrap_schedule={self.ex_schedule!r}")

    def _block_order(self) -> list[int]:
        B = len(self.ex_bounds)
        if self.ex_order == "stability" and self._score_ema is not None and self.score_name == "direction_stability":
            l2b = layer_to_block(self.L, self.ex_bounds)
            per_block = torch.zeros(B, dtype=torch.float64)
            cnt = torch.zeros(B, dtype=torch.float64)
            for l in range(self.L):
                per_block[l2b[l]] += self._score_ema[l]
                cnt[l2b[l]] += 1
            return torch.argsort(per_block / cnt.clamp(min=1), descending=True).tolist()
        if self.ex_order == "bottom_up":
            return list(range(B))
        if self.ex_order == "top_down":
            return list(range(B))[::-1]
        # middle_out (default, and the fallback for 'stability' before any score exists)
        center = (B - 1) / 2.0
        return sorted(range(B), key=lambda b: (abs(b - center), -b))

    @torch.no_grad()
    def _run_block_extrapolation(self, data):
        # mark this event as done and schedule the next one
        if self.ex_schedule == "exp":
            self._next_extrap_step = max(2 * self._next_extrap_step, self._next_extrap_step + 1)
        self._last_extrap_step = self.step
        B = len(self.ex_bounds)
        J0, logp0, mask = self._probe(data)
        J_best = J0
        accepted = torch.zeros(B, dtype=torch.float64)
        kl_last = 0.0
        for b in self._block_order():
            params = self._block_params(self.ex_bounds[b], self.ex_scope)
            orig = {name: self._to_cpu(p) for name, p in params}
            delta = {name: orig[name] - self._w_ckpt[name] for name, p in params}  # W_t - W_prev (CPU fp32)
            dnorm = math.sqrt(sum(d.pow(2).sum().item() for d in delta.values()))
            best_alpha = 0.0
            for alpha in self.ex_alphas:
                for name, p in params:
                    p.copy_((orig[name] + alpha * delta[name]).to(p.device))
                J, logp, _ = self._probe(data)
                kl = _k3_kl(logp0, logp, mask)
                ok = (J >= J_best - self.ex_tol) and (kl <= self.ex_max_kl)
                if self.rank == 0:
                    print(f"[LayerLR] extrap step={self.step} block={b} alpha={alpha} J={J:.5f} (best {J_best:.5f}) kl={kl:.4e} -> {'accept' if ok else 'reject'}")
                if ok:
                    J_best, best_alpha, kl_last = J, alpha, kl
                else:
                    break
            for name, p in params:  # leave the block at the best accepted alpha (0 = unchanged)
                p.copy_((orig[name] + best_alpha * delta[name]).to(p.device))
            accepted[b] = best_alpha
            self._metrics[f"layer_lr/extrap_alpha/b{b}"] = best_alpha
            self._metrics[f"layer_lr/extrap_dnorm/b{b}"] = dnorm
            del orig, delta
        self._metrics["layer_lr/extrap/J_before"] = J0
        self._metrics["layer_lr/extrap/J_after"] = J_best
        self._metrics["layer_lr/extrap/kl"] = kl_last
        self._metrics["layer_lr/extrap/events"] = self._metrics.get("layer_lr/extrap/events", 0.0) + 1.0
        # the next local direction is measured from the (possibly extrapolated) current weights
        self._w_ckpt = self._snapshot(self._block_entries(self.ex_scope))
        if self.ex_feedback:
            amax = max(self.ex_alphas) if self.ex_alphas else 1.0
            l2b = layer_to_block(self.L, self.ex_bounds)
            m = torch.tensor([1.0 + (accepted[l2b[l]].item() / amax) * (self.max_mult - 1.0) for l in range(self.L)], dtype=torch.float64)
            if self._ex_mult is None:
                self._ex_mult = m
            else:
                self._ex_mult.mul_(self.ema_beta).add_(m, alpha=1.0 - self.ema_beta)
        if self.rank == 0:
            print(f"[LayerLR] extrap step={self.step} accepted alphas={accepted.tolist()} J {J0:.5f} -> {J_best:.5f}")
