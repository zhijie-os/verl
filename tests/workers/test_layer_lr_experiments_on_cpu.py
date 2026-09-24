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
"""CPU tests for verl/workers/engine/fsdp/layer_lr_experiments.py.

Run from the repo root inside the VERL environment (no GPU needed):

    python -m pytest tests/workers/test_layer_lr_experiments_on_cpu.py -x -q

A tiny Qwen-like module (model.layers.<i>.self_attn / .mlp / norms, embed_tokens, lm_head)
and a fake engine stand in for FSDP.  The probe forward pass is replaced by closed-form
objectives so that the leave-one-block-out utilities and the extrapolation accept/reject logic
can be checked exactly; the real probe (nested-tensor plumbing) is tested separately when
tensordict + verl are importable.
"""

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = _REPO / "verl" / "workers" / "engine" / "fsdp" / "layer_lr_experiments.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("layer_lr_experiments", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lle = _load_module()


# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------
class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d // 2, bias=True)
        self.v_proj = nn.Linear(d, d // 2, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)


class _MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)


class _Block(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, ff)
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)


class _Inner(nn.Module):
    def __init__(self, L, d, ff, V):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, d)
        self.layers = nn.ModuleList([_Block(d, ff) for _ in range(L)])
        self.norm = nn.LayerNorm(d)


class TinyLM(nn.Module):
    def __init__(self, L=6, d=8, ff=16, V=11):
        super().__init__()
        self.model = _Inner(L, d, ff, V)
        self.lm_head = nn.Linear(d, V, bias=False)


class FakeEngine:
    def __init__(self, module, cfg, L):
        self.optimizer_config = cfg
        self.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=L))
        self.rank = 0
        self.module = module
        self.optimizer = None
        self.lr_scheduler = None
        self.fake_log_probs = None

    def forward_backward_batch(self, data, loss_function, forward_only):
        assert forward_only and loss_function is None
        return {"model_output": {"log_probs": self.fake_log_probs}, "loss": [1.0], "metrics": {}}


_DEFAULTS = dict(
    lr=5e-7,
    weight_decay=0.0,
    betas=(0.9, 0.999),
    clip_grad=1.0,
    topk_svd_enabled=False,
    layerwise_lr_enabled=False,
    middle_lr_multiplier=2.0,
    middle_layer_start_frac=1.0 / 3.0,
    middle_layer_end_frac=2.0 / 3.0,
    adaptive_layer_lr_enabled=False,
    adaptive_layer_lr_interval=10,
    adaptive_layer_lr_topk_ratio=0.10,
    adaptive_layer_lr_max_multiplier=2.0,
    adaptive_layer_lr_ema_beta=0.90,
    adaptive_layer_lr_score="topk_energy",
    adaptive_layer_lr_score_scope="all",
    adaptive_layer_lr_apply_scope="all",
    adaptive_layer_lr_mapping="minmax",
    adaptive_layer_lr_prior="none",
    adaptive_layer_lr_gamma=0.25,
    adaptive_layer_lr_target_mean=-1.0,
    adaptive_layer_lr_min_multiplier=0.5,
    adaptive_layer_lr_fit_profile=False,
    adaptive_layer_lr_stability_ref="prev_window",
    adaptive_layer_lr_z_floor=0.0,
    layerwise_profile="window",
    layerwise_center_frac=0.5,
    layerwise_width_frac=1.0 / 6.0,
    layerwise_module_scope="all",
    periphery_lr_multiplier=1.0,
    other_lr_multiplier=1.0,
    utility_probe_enabled=False,
    utility_probe_interval=20,
    utility_probe_num_blocks=4,
    utility_probe_module_scope="all",
    utility_probe_mode="leave_one_out",
    utility_probe_ema_beta=0.7,
    utility_probe_normalize_by_norm=True,
    utility_probe_gamma=0.5,
    probe_objective="adv_logp",
    block_extrap_enabled=False,
    block_extrap_schedule="exp",
    block_extrap_interval=50,
    block_extrap_first_step=1,
    block_extrap_max_step=-1,
    block_extrap_num_blocks=4,
    block_extrap_module_scope="all",
    block_extrap_alphas="1,2,4,8",
    block_extrap_tolerance=0.0,
    block_extrap_max_kl=0.05,
    block_extrap_order="middle_out",
    block_extrap_feedback_lr=False,
)


def make_cfg(**overrides):
    """The real FSDPOptimizerConfig when verl is importable (validates the new fields), else a namespace."""
    values = dict(_DEFAULTS)
    values.update(overrides)
    try:
        from verl.workers.config.optimizer import FSDPOptimizerConfig
    except ImportError:  # outside the verl environment: a plain namespace
        return SimpleNamespace(**values)
    return FSDPOptimizerConfig(**values)  # inside verl: every field must exist on the dataclass


def make_setup(L=6, **overrides):
    torch.manual_seed(0)
    model = TinyLM(L=L)
    cfg = make_cfg(**overrides)
    engine = FakeEngine(model, cfg, L)
    ctrl = lle.LayerLRController(engine, model)
    if ctrl.enabled:
        groups = ctrl.build_param_groups(model)
    else:
        groups = list(model.parameters())
    engine.optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=0.0)
    engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda s: 1.0)
    return model, engine, ctrl


def _layer_params(model, l):
    return [(n, p) for n, p in model.named_parameters() if f"model.layers.{l}." in n]


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def test_helpers():
    assert lle.param_kind("model.layers.3.mlp.up_proj.weight") == "mlp"
    assert lle.param_kind("_fsdp_wrapped_module.model.layers.3._fsdp_wrapped_module.self_attn.q_proj.weight") == "attn"
    assert lle.param_kind("model.layers.3.input_layernorm.weight") == "other"
    assert lle.param_kind("model.embed_tokens.weight") == "other"
    assert lle.parse_float_list("1, 2,4") == [1.0, 2.0, 4.0]
    assert lle.parse_float_list([2, 4]) == [2.0, 4.0]
    assert lle.block_bounds(28, 4) == [(0, 7), (7, 14), (14, 21), (21, 28)]
    assert lle.block_bounds(6, 3) == [(0, 2), (2, 4), (4, 6)]
    assert lle.layer_to_block(6, lle.block_bounds(6, 3)) == [0, 0, 1, 1, 2, 2]
    # window == original layerwise run for 28 layers: layers 9..18
    w = lle.depth_bump(28, "window", 1 / 3, 2 / 3, 0.5, 1 / 6)
    assert [int(x) for x in w.tolist()] == [1 if 9 <= l <= 18 else 0 for l in range(28)]
    g = lle.depth_bump(28, "gaussian", 1 / 3, 2 / 3, 0.5, 1 / 6)
    assert g.argmax().item() in (13, 14) and g[0].item() < 0.05
    assert abs(g[13].item() - g[14].item()) < 1e-9  # symmetric around 13.5
    c = lle.depth_bump(28, "cosine", 1 / 3, 2 / 3, 0.5, 1 / 6)
    # the centre 13.5 sits between two layers, so the peak is 0.5 * (1 + cos(pi * 0.5 / w)) = 0.972, not 1.0
    peak = 0.5 * (1.0 + math.cos(math.pi * 0.5 / (28 / 6)))
    assert c[0].item() == 0.0 and abs(c.max().item() - peak) < 1e-9 and c.argmax().item() in (13, 14)


def test_fit_gaussian_bump_recovers_center_and_width():
    torch.manual_seed(1)
    L = 28
    l = torch.arange(L, dtype=torch.float64)
    truth = 0.3 + 1.2 * torch.exp(-0.5 * ((l - 14.0) / 4.0) ** 2)
    noisy = truth + 0.05 * torch.randn(L, dtype=torch.float64)
    curve, params = lle.fit_gaussian_bump(noisy)
    assert abs(params["c"] - 14.0) <= 1.0
    assert params["w"] in (3.0, 4.0, 6.0)
    assert (curve - truth).abs().max().item() < 0.15


# ------------------------------------------------------------------------------------------------
# fixed profiles
# ------------------------------------------------------------------------------------------------
def test_inert_when_disabled():
    model, engine, ctrl = make_setup()
    assert not ctrl.enabled
    ctrl.on_optimizer_step()
    ctrl.on_train_batch_begin(None)
    ctrl.on_lr_scheduler_step()
    assert ctrl.pop_metrics() == {}
    assert len(engine.optimizer.param_groups) == 1


def test_window_profile_reproduces_original_layerwise_run():
    L = 28
    model, engine, ctrl = make_setup(L=L, layerwise_lr_enabled=True, middle_lr_multiplier=2.0)
    assert ctrl.enabled
    # one group per (layer, kind) + one for embeddings/lm_head/final norm
    kinds = {(g["layer_idx"], g["kind"]) for g in engine.optimizer.param_groups}
    assert (-1, "other") in kinds and (0, "attn") in kinds and (27, "mlp") in kinds and (5, "other") in kinds
    n_params_groups = sum(len(g["params"]) for g in engine.optimizer.param_groups)
    assert n_params_groups == len(list(model.parameters()))
    ctrl.on_optimizer_step()  # applies the multipliers
    for g in engine.optimizer.param_groups:
        l = g["layer_idx"]
        expected = 2.0 if 9 <= l <= 18 else 1.0
        assert math.isclose(g["lr"], 5e-7 * expected, rel_tol=1e-9), (l, g["kind"], g["lr"])
    m = ctrl.pop_metrics()
    assert m["layer_lr/mult/l09"] == 2.0 and m["layer_lr/mult/l08"] == 1.0 and m["layer_lr/mult/l19"] == 1.0
    # the scheduler overwrites group lrs; the hook must restore the profile
    engine.lr_scheduler.step()
    ctrl.on_lr_scheduler_step()
    for g in engine.optimizer.param_groups:
        expected = 2.0 if 9 <= g["layer_idx"] <= 18 else 1.0
        assert math.isclose(g["lr"], 5e-7 * expected, rel_tol=1e-9)


def test_mlp_scope_periphery_and_frozen_other():
    L = 6  # window: floor(6/3)=2 .. ceil(12/3)=4 -> layers 2, 3
    model, engine, ctrl = make_setup(
        L=L,
        layerwise_lr_enabled=True,
        middle_lr_multiplier=3.0,
        layerwise_module_scope="mlp",
        periphery_lr_multiplier=0.5,
        other_lr_multiplier=0.0,
    )
    ctrl.on_optimizer_step()
    for g in engine.optimizer.param_groups:
        l, kind = g["layer_idx"], g["kind"]
        if l < 0:
            expected = 0.0
        elif kind == "mlp":
            expected = 3.0 if l in (2, 3) else 0.5
        else:
            expected = 1.0  # attention / norms untouched by an MLP-only profile
        assert math.isclose(g["lr"], 5e-7 * expected, rel_tol=1e-9), (l, kind, g["lr"])
    # lr = 0 must leave frozen parameters untouched by AdamW, while boosted ones move
    before_head = model.lm_head.weight.detach().clone()
    before_mlp = model.model.layers[2].mlp.up_proj.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    engine.optimizer.step()
    assert torch.equal(model.lm_head.weight.detach(), before_head)
    assert not torch.equal(model.model.layers[2].mlp.up_proj.weight.detach(), before_mlp)


def test_gaussian_profile_peaks_in_the_middle():
    L = 28
    model, engine, ctrl = make_setup(L=L, layerwise_lr_enabled=True, layerwise_profile="gaussian", middle_lr_multiplier=2.0)
    ctrl.on_optimizer_step()
    m = ctrl.pop_metrics()
    mults = [m[f"layer_lr/mult/l{l:02d}"] for l in range(L)]
    assert max(mults) > 1.95 and mults.index(max(mults)) in (13, 14)
    assert mults[0] < 1.05 and mults[27] < 1.05


# ------------------------------------------------------------------------------------------------
# adaptive scores
# ------------------------------------------------------------------------------------------------
def test_topk_energy_minmax_matches_original_mapping():
    L = 6
    model, engine, ctrl = make_setup(
        L=L, adaptive_layer_lr_enabled=True, adaptive_layer_lr_interval=1, adaptive_layer_lr_max_multiplier=2.0
    )
    for l in range(L):
        scale = 10.0 if l == 4 else 1.0
        for n, p in _layer_params(model, l):
            p.grad = scale * torch.randn_like(p)
    ctrl.on_optimizer_step()
    mult = ctrl.layer_mult
    assert mult is not None and mult.argmax().item() == 4
    assert math.isclose(mult.max().item(), 2.0, abs_tol=1e-9) and math.isclose(mult.min().item(), 1.0, abs_tol=1e-9)
    for g in engine.optimizer.param_groups:
        if g["layer_idx"] == 4:
            assert math.isclose(g["lr"], 1e-6, rel_tol=1e-9)


def test_direction_stability_scores_and_prior_exp_mapping():
    L = 6
    model, engine, ctrl = make_setup(
        L=L,
        adaptive_layer_lr_enabled=True,
        adaptive_layer_lr_interval=1,
        adaptive_layer_lr_score="direction_stability",
        adaptive_layer_lr_mapping="prior_exp",
        adaptive_layer_lr_prior="none",
        adaptive_layer_lr_gamma=0.5,
        adaptive_layer_lr_target_mean=1.0,
        adaptive_layer_lr_min_multiplier=0.5,
        adaptive_layer_lr_max_multiplier=3.0,
        adaptive_layer_lr_ema_beta=0.0,
    )
    torch.manual_seed(3)
    directions = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    with torch.no_grad():
        for step in range(4):
            for l in range(L):
                for n, p in _layer_params(model, l):
                    if l < 3:  # consistent drift
                        p.add_(1e-3 * directions[n])
                    else:  # fresh noise every window
                        p.add_(1e-3 * torch.randn_like(p))
            ctrl.on_optimizer_step()
    cos = ctrl._score_ema
    assert cos is not None
    assert all(cos[l].item() > 0.99 for l in range(3)), cos
    assert all(abs(cos[l].item()) < 0.3 for l in range(3, 6)), cos
    mult = ctrl.layer_mult
    assert all(mult[l].item() > mult[k].item() for l in range(3) for k in range(3, 6))
    assert math.isclose(mult.mean().item(), 1.0, rel_tol=0.05)  # budget-neutral
    m = ctrl.pop_metrics()
    assert "layer_lr/score/l00" in m and "layer_lr/mult/l05" in m


def test_direction_stability_cumulative_reference_and_z_floor():
    L = 6
    model, engine, ctrl = make_setup(
        L=L,
        adaptive_layer_lr_enabled=True,
        adaptive_layer_lr_interval=1,
        adaptive_layer_lr_score="direction_stability",
        adaptive_layer_lr_stability_ref="cumulative",
        adaptive_layer_lr_mapping="prior_exp",
        adaptive_layer_lr_gamma=0.5,
        adaptive_layer_lr_target_mean=1.0,
        adaptive_layer_lr_max_multiplier=3.0,
        adaptive_layer_lr_ema_beta=0.0,
    )
    torch.manual_seed(4)
    directions = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    with torch.no_grad():
        for step in range(3):
            for l in range(L):
                for n, p in _layer_params(model, l):
                    if l < 3:
                        p.add_(1e-3 * directions[n])  # keeps moving along its accumulated direction
                    else:
                        p.add_(1e-3 * torch.randn_like(p))  # wanders
            ctrl.on_optimizer_step()
    # scored from the 2nd measurement on: window 2 vs the accumulated window 1
    cos = ctrl._score_ema
    assert cos is not None
    assert all(cos[l].item() > 0.99 for l in range(3)), cos
    assert all(abs(cos[l].item()) < 0.3 for l in range(3, 6)), cos
    assert all(ctrl.layer_mult[l].item() > ctrl.layer_mult[k].item() for l in range(3) for k in range(3, 6))

    # z floor: a flat profile must not be amplified into +-2
    flat = torch.tensor([0.50, 0.51, 0.50, 0.49, 0.50, 0.51], dtype=torch.float64)
    assert lle.standardize(flat).abs().max().item() > 1.0  # plain z-score: noise looks like signal
    assert lle.standardize(flat, floor=0.05).abs().max().item() < 0.25  # floored: stays near the prior


def test_layerwise_plus_adaptive_uses_profile_as_prior_until_first_score():
    L = 6
    model, engine, ctrl = make_setup(
        L=L,
        layerwise_lr_enabled=True,
        middle_lr_multiplier=2.0,
        adaptive_layer_lr_enabled=True,
        adaptive_layer_lr_interval=100,
        adaptive_layer_lr_score="direction_stability",
        adaptive_layer_lr_mapping="prior_exp",
    )
    assert ctrl.prior_name == "profile"
    ctrl.on_optimizer_step()
    for g in engine.optimizer.param_groups:
        l = g["layer_idx"]
        expected = 1.0 if l < 0 else (2.0 if l in (2, 3) else 1.0)
        assert math.isclose(g["lr"], 5e-7 * expected, rel_tol=1e-9)


# ------------------------------------------------------------------------------------------------
# utility probe (closed-form objective in place of the forward pass)
# ------------------------------------------------------------------------------------------------
def _install_linear_probe(ctrl, model, coeffs):
    """J(W) = sum_l coeffs[l] * sum(W_l - W0_l) over the layer's parameters (per-layer utility ~ coeffs)."""
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}

    def probe(data):
        J = 0.0
        for n, p in model.named_parameters():
            m = lle._LAYER_RE.search(n)
            if m is None:
                continue
            J += coeffs[int(m.group(1))] * (p.detach() - w0[n]).sum().item()
        return J, torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool)

    ctrl._probe = probe
    return w0


def test_utility_probe_leave_one_out_orders_blocks_and_restores_weights():
    L = 6
    model, engine, ctrl = make_setup(
        L=L,
        utility_probe_enabled=True,
        utility_probe_interval=5,
        utility_probe_num_blocks=3,
        utility_probe_normalize_by_norm=False,
        utility_probe_gamma=0.5,
        utility_probe_ema_beta=0.0,
        adaptive_layer_lr_target_mean=1.0,
        adaptive_layer_lr_max_multiplier=3.0,
    )
    coeffs = [1.0, 1.0, 3.0, 3.0, -1.0, -1.0]  # block utilities: 2 * 1, 2 * 3, 2 * (-1) (per unit shift)
    w0 = _install_linear_probe(ctrl, model, coeffs)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(0.01)  # same shift everywhere -> utility ordering is set by coeffs
    shifted = {n: p.detach().clone() for n, p in model.named_parameters()}
    ctrl.step = 5
    ctrl.on_train_batch_begin(None)
    m = ctrl.pop_metrics()
    u = [m[f"layer_lr/utility/b{b}"] for b in range(3)]
    assert u[1] > u[0] > u[2], u
    for n, p in model.named_parameters():  # weights restored exactly after the probe
        assert torch.equal(p.detach(), shifted[n]), n
    mult = ctrl.layer_mult
    assert mult[2].item() > mult[0].item() > mult[4].item()
    assert math.isclose(mult.mean().item(), 1.0, rel_tol=0.05)
    # the probe does not re-run for the same optimizer step
    def must_not_run(data):
        raise AssertionError("probe must not run twice for the same optimizer step")

    ctrl._probe = must_not_run
    ctrl.on_train_batch_begin(None)


def test_utility_probe_inject_mode_matches_leave_one_out_for_linear_objective():
    L = 6
    model, engine, ctrl = make_setup(
        L=L, utility_probe_enabled=True, utility_probe_interval=1, utility_probe_num_blocks=3, utility_probe_mode="inject",
        utility_probe_normalize_by_norm=False, utility_probe_ema_beta=0.0,
    )
    coeffs = [1.0, 1.0, 3.0, 3.0, -1.0, -1.0]
    _install_linear_probe(ctrl, model, coeffs)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(0.01)
    shifted = {n: p.detach().clone() for n, p in model.named_parameters()}
    ctrl.step = 1
    ctrl.on_train_batch_begin(None)
    m = ctrl.pop_metrics()
    u = [m[f"layer_lr/utility/b{b}"] for b in range(3)]
    assert u[1] > u[0] > u[2], u
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), shifted[n]), n


# ------------------------------------------------------------------------------------------------
# block-wise extrapolation
# ------------------------------------------------------------------------------------------------
def test_block_extrapolation_accepts_best_alpha_and_updates_checkpoint():
    L = 6
    model, engine, ctrl = make_setup(
        L=L,
        block_extrap_enabled=True,
        block_extrap_schedule="exp",
        block_extrap_first_step=1,
        block_extrap_num_blocks=3,
        block_extrap_alphas="1,2,4",
        block_extrap_max_kl=1.0,
    )
    torch.manual_seed(5)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    delta = {n: 1e-2 * torch.randn_like(p) for n, p in model.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(delta[n])  # W_t = W_0 + delta

    # J(W) = -|| W - (W_0 + 3 delta) ||^2  -> best candidate is alpha = 2 (W_t + 2 delta), alpha = 4 overshoots
    def probe(data):
        J = 0.0
        for n, p in model.named_parameters():
            if lle._LAYER_RE.search(n) is None:
                continue
            J -= (p.detach() - (w0[n] + 3.0 * delta[n])).pow(2).sum().item()
        return J, torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool)

    ctrl._probe = probe
    ctrl.step = 1
    assert ctrl._extrap_due()
    ctrl.on_train_batch_begin(None)
    m = ctrl.pop_metrics()
    for b in range(3):
        assert m[f"layer_lr/extrap_alpha/b{b}"] == 2.0, m
    for n, p in model.named_parameters():
        if lle._LAYER_RE.search(n) is None:
            assert torch.equal(p.detach(), w0[n] + delta[n])  # non-layer params untouched
        else:
            assert torch.allclose(p.detach(), w0[n] + 3.0 * delta[n], atol=1e-6), n
    assert ctrl._next_extrap_step == 2 and not ctrl._extrap_due()
    ctrl.step = 2
    assert ctrl._extrap_due()
    # the checkpoint moved to the extrapolated weights
    for n, l, kind, _ in ctrl._layer_entries:
        assert torch.allclose(ctrl._w_ckpt[n], w0[n] + 3.0 * delta[n], atol=1e-6)
    assert m["layer_lr/extrap/J_after"] >= m["layer_lr/extrap/J_before"]


def test_block_extrapolation_kl_guard_rejects():
    L = 6
    model, engine, ctrl = make_setup(
        L=L, block_extrap_enabled=True, block_extrap_first_step=1, block_extrap_num_blocks=2, block_extrap_alphas="1,2",
        block_extrap_max_kl=1e-6,
    )
    w = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(1e-2)
    calls = {"n": 0}

    def probe(data):  # J always improves, but the policy moves too much -> every alpha rejected
        calls["n"] += 1
        logp = torch.full((2, 3), -1.0) - 0.5 * (calls["n"] - 1)
        return float(calls["n"]), logp, torch.ones(2, 3, dtype=torch.bool)

    ctrl._probe = probe
    ctrl.step = 1
    ctrl.on_train_batch_begin(None)
    m = ctrl.pop_metrics()
    assert m["layer_lr/extrap_alpha/b0"] == 0.0 and m["layer_lr/extrap_alpha/b1"] == 0.0
    for n, p in model.named_parameters():
        assert torch.allclose(p.detach(), w[n] + 1e-2)


def test_block_extrapolation_feedback_lr_and_interval_schedule():
    L = 6
    model, engine, ctrl = make_setup(
        L=L, block_extrap_enabled=True, block_extrap_schedule="interval", block_extrap_interval=10,
        block_extrap_num_blocks=2, block_extrap_alphas="1,2", block_extrap_feedback_lr=True,
        adaptive_layer_lr_max_multiplier=3.0, adaptive_layer_lr_ema_beta=0.0, block_extrap_max_kl=1.0,
    )
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(1e-2)
    ctrl._probe = lambda data: (1.0, torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool))  # flat J: all alphas accepted
    ctrl.step = 10
    assert ctrl._extrap_due()
    ctrl.on_train_batch_begin(None)
    assert not ctrl._extrap_due()  # same step: not again
    m = ctrl.pop_metrics()
    assert m["layer_lr/extrap_alpha/b0"] == 2.0
    # alpha_max accepted everywhere -> every layer at the max multiplier
    for g in engine.optimizer.param_groups:
        if g["layer_idx"] >= 0:
            assert math.isclose(g["lr"], 5e-7 * 3.0, rel_tol=1e-9)
    ctrl.step = 20
    assert ctrl._extrap_due()


# ------------------------------------------------------------------------------------------------
# the real probe objective (nested-tensor plumbing) -- needs tensordict + verl
# ------------------------------------------------------------------------------------------------
def test_probe_objective_with_nested_batch():
    pytest.importorskip("tensordict")
    pytest.importorskip("verl")
    from tensordict import TensorDict

    L = 2
    model, engine, ctrl = make_setup(L=L, utility_probe_enabled=True)
    prompts = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    responses = torch.nested.as_nested_tensor([torch.tensor([6, 7, 8, 9]), torch.tensor([10, 11])], layout=torch.jagged)
    response_mask = torch.nested.as_nested_tensor([torch.tensor([1, 1, 1, 0]), torch.tensor([1, 1])], layout=torch.jagged)
    advantages = torch.nested.as_nested_tensor(
        [torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([-1.0, -1.0])], layout=torch.jagged
    )
    data = TensorDict(
        {"prompts": prompts, "responses": responses, "response_mask": response_mask, "advantages": advantages},
        batch_size=[2],
    )
    # full-sequence log-probs: seq lengths 7 and 4; response log-probs are values[off - R - 1 : off - 1]
    lp0 = torch.arange(7, dtype=torch.float32) * -0.1  # response part: indices 2..5 -> -0.2, -0.3, -0.4, -0.5
    lp1 = torch.arange(4, dtype=torch.float32) * -0.2  # response part: indices 1..2 -> -0.2, -0.4
    engine.fake_log_probs = torch.nested.as_nested_tensor([lp0, lp1], layout=torch.jagged)
    J, logp, mask = ctrl._probe(data)
    expected = (0.5 * (-0.2 - 0.3 - 0.4) + (-1.0) * (-0.2 - 0.4)) / (0.5 * 3 + 1.0 * 2)
    assert math.isclose(J, expected, rel_tol=1e-5), (J, expected)
    assert logp.shape == (2, 4) and mask.shape == (2, 4)
    assert mask.sum().item() == 5
