# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components._peft.lora_experts_nf4 import GroupedExpertsLoRANF4
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import GroupedExperts
from nemo_automodel.components.moe.quantized_experts import (
    GroupedExpertsNF4,
    apply_nf4_to_moe_experts,
    assert_nf4_placeholders_on_meta,
    load_nf4_experts_from_hf_checkpoint,
    nf4_placeholders_pending,
)
from nemo_automodel.components.quantization.nf4 import NF4_BLOCK_SIZE, dequantize_nf4, quantize_nf4
from nemo_automodel.shared.parameter_names import SKIP_CHECKPOINT_LOAD_ATTR

BASE_NAMES = ("gate_and_up_projs", "down_projs")
PACKED_NAMES = {"gate_and_up_projs_nf4", "gate_and_up_projs_absmax", "down_projs_nf4", "down_projs_absmax"}
LORA_NAMES = {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    # Both contraction dims divisible by the NF4 block (64) and grouped-GEMM aligned.
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


def _grouped_mm_available(device: torch.device) -> bool:
    x = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
    w = torch.randn(2, 64, 128, dtype=torch.bfloat16, device=device)
    try:
        torch._grouped_mm(x, w, offs=torch.tensor([2, 4], dtype=torch.int32, device=device))
    except (RuntimeError, NotImplementedError):
        return False
    return True


def _materialized_experts(moe_config, device) -> GroupedExperts:
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    return orig


def _make_representable_(experts: GroupedExperts) -> None:
    """Replace expert weights in-place with their NF4 round-trip so the packed storage is exact."""
    with torch.no_grad():
        for name in BASE_NAMES:
            param = getattr(experts, name)
            codes, absmax = quantize_nf4(param.data.transpose(-2, -1).contiguous())
            param.data.copy_(dequantize_nf4(codes, absmax, param.dtype).transpose(-2, -1))


def _routing_inputs(moe_config, num_tokens, device, dtype):
    torch.manual_seed(7)
    x = torch.randn(num_tokens, moe_config.dim, dtype=dtype, device=device, requires_grad=True)
    indices = torch.stack(
        [
            torch.randperm(moe_config.n_routed_experts, device=device)[: moe_config.n_activated_experts]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=dtype, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return x, token_mask, weights, indices


def test_nf4_module_packs_base_weights(moe_config, device):
    lora = GroupedExpertsLoRANF4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16)

    assert lora._nf4_resident
    assert not hasattr(lora, "gate_and_up_projs") and not hasattr(lora, "down_projs")
    assert lora.gate_and_up_projs_nf4.dtype == torch.uint8
    assert lora.gate_and_up_projs_absmax.dtype == torch.float32
    assert not lora.gate_and_up_projs_nf4.requires_grad
    # Checkpoint orientation [E, out, in]: codes pack the contraction dim, absmax has one value per block.
    assert lora.gate_and_up_projs_nf4.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    assert lora.gate_and_up_projs_absmax.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // NF4_BLOCK_SIZE)
    assert lora.down_projs_nf4.shape == (4, moe_config.dim, moe_config.moe_inter_dim // 2)
    assert {n for n, p in lora.named_parameters() if p.requires_grad} == LORA_NAMES
    # Materialized weights were quantized at construction: nothing is left to stream.
    assert not nf4_placeholders_pending(lora)


def test_nf4_forward_backward_matches_bf16(moe_config, device):
    """With NF4-representable base weights, the packed module must match the bf16 module."""
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    orig = _materialized_experts(moe_config, device)
    _make_representable_(orig)

    ref = GroupedExpertsLoRA(orig, lora_dim=8, alpha=16).to(device)
    q = GroupedExpertsLoRANF4(orig, lora_dim=8, alpha=16).to(device)
    with torch.no_grad():
        for name in LORA_NAMES:
            getattr(q, name).data.copy_(getattr(ref, name).data)

    x_ref, token_mask, weights, indices = _routing_inputs(moe_config, 32, device, torch.bfloat16)
    x_q = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref, token_mask, weights, indices)
    y_q = q(x_q, token_mask, weights, indices)
    torch.testing.assert_close(y_q, y_ref, atol=1e-6, rtol=1e-6)

    y_ref.float().pow(2).sum().backward()
    y_q.float().pow(2).sum().backward()
    torch.testing.assert_close(x_q.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for name in LORA_NAMES:
        torch.testing.assert_close(getattr(q, name).grad, getattr(ref, name).grad, atol=1e-5, rtol=1e-5)
    assert q.gate_and_up_projs_nf4.grad is None and q.down_projs_nf4.grad is None


def test_nf4_no_routed_tokens(moe_config, device):
    """The all-masked dummy path must run and keep LoRA gradients flowing."""
    q = GroupedExpertsLoRANF4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16).to(device)
    x, _, weights, indices = _routing_inputs(moe_config, 8, device, torch.bfloat16)
    token_mask = torch.zeros(8, dtype=torch.bool, device=device)

    y = q(x, token_mask, weights, indices)
    assert y.shape == x.shape
    y.float().sum().backward()
    assert x.grad is not None


def test_nf4_requires_torch_mm_backend(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)  # default per-expert loop backend
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsLoRANF4(orig, lora_dim=8, alpha=16)


def test_patch_moe_module_selects_nf4(moe_config, device):
    orig = _materialized_experts(moe_config, device)
    patched = patch_moe_module(orig, dim=8, alpha=16, expert_weight_format="nf4")
    assert isinstance(patched, GroupedExpertsLoRANF4)
    with pytest.raises(ValueError, match="'nf4'"):
        patch_moe_module(orig, dim=8, alpha=16, expert_weight_format="int8")


def test_frozen_passthrough_init_registers_placeholders_on_meta(moe_config):
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
    orig.use_torch_mm = True

    q = GroupedExpertsNF4(orig)

    assert q._nf4_resident
    assert not hasattr(q, "gate_and_up_projs")
    assert q.gate_and_up_projs_nf4.is_meta and q.gate_and_up_projs_nf4.dtype == torch.uint8
    assert q.down_projs_absmax.is_meta and q.down_projs_absmax.dtype == torch.float32
    assert q.gate_and_up_projs_nf4.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    assert [n for n, p in q.named_parameters() if p.requires_grad] == []
    # Placeholders are flagged for the module-owned stream, not for the DCP base load.
    assert getattr(q, SKIP_CHECKPOINT_LOAD_ATTR) == PACKED_NAMES
    assert nf4_placeholders_pending(q)
    assert_nf4_placeholders_on_meta(q)


def test_lora_passthrough_init_placeholders_with_trainable_adapters(moe_config):
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
        orig.use_torch_mm = True
        q = GroupedExpertsLoRANF4(orig, lora_dim=8, alpha=16)

    assert q._nf4_resident
    assert not hasattr(q, "gate_and_up_projs")
    assert q.gate_and_up_projs_nf4.is_meta
    assert {n for n, p in q.named_parameters() if p.requires_grad} == LORA_NAMES
    assert getattr(q, SKIP_CHECKPOINT_LOAD_ATTR) == PACKED_NAMES


def test_apply_nf4_to_moe_experts_converts_remaining_frozen_experts(moe_config, device):
    model = nn.Module()
    model.lora_experts = GroupedExpertsLoRANF4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16)
    model.frozen_experts = _materialized_experts(moe_config, device)
    model.linear = nn.Linear(4, 4)

    apply_nf4_to_moe_experts(model)

    assert isinstance(model.lora_experts, GroupedExpertsLoRANF4)
    assert isinstance(model.frozen_experts, GroupedExpertsNF4)
    assert isinstance(model.linear, nn.Linear)
    assert [n for n, p in model.frozen_experts.named_parameters() if p.requires_grad] == []


def _write_hf_checkpoint(experts: GroupedExperts, inter: int, checkpoint_dir) -> None:
    """Write the stacked expert weights in the per-expert HF layout ([out, in] matrices)."""
    from safetensors.torch import save_file

    tensors = {}
    gate_and_up, down = experts.gate_and_up_projs.data, experts.down_projs.data  # [E, in, 2*inter], [E, inter, out]
    for e in range(gate_and_up.shape[0]):
        base = f"model.layers.3.mlp.experts.{e}"
        tensors[f"{base}.gate_proj.weight"] = gate_and_up[e, :, :inter].T.contiguous()
        tensors[f"{base}.up_proj.weight"] = gate_and_up[e, :, inter:].T.contiguous()
        tensors[f"{base}.down_proj.weight"] = down[e].T.contiguous()
    save_file(tensors, str(checkpoint_dir / "model.safetensors"))


def test_stream_placeholders_from_hf_checkpoint(moe_config, tmp_path):
    source = _materialized_experts(moe_config, torch.device("cpu"))
    _write_hf_checkpoint(source, moe_config.moe_inter_dim, tmp_path)

    # The HF layer index comes from the module path, so the experts live at layers.3.
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
    model.layers[3].mlp = nn.Module()
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
        orig.use_torch_mm = True
        model.layers[3].mlp.experts = GroupedExpertsLoRANF4(orig, lora_dim=8, alpha=16)
    experts = model.layers[3].mlp.experts
    assert nf4_placeholders_pending(model)

    stats = load_nf4_experts_from_hf_checkpoint(model, str(tmp_path), device=torch.device("cpu"))

    assert stats["modules"] == 1 and stats["experts"] == moe_config.n_routed_experts
    assert not nf4_placeholders_pending(model) and not hasattr(experts, SKIP_CHECKPOINT_LOAD_ATTR)
    for name in BASE_NAMES:
        codes, absmax = quantize_nf4(getattr(source, name).data.transpose(-2, -1).contiguous())
        codes_name, absmax_name = experts.nf4_parameter_names(name)
        assert torch.equal(getattr(experts, codes_name), codes)
        assert torch.equal(getattr(experts, absmax_name), absmax)
        assert not getattr(experts, codes_name).requires_grad
    # The LoRA adapters are untouched by the stream.
    assert experts.lora_gate_and_up_A.is_meta


def test_stream_rejects_materialized_placeholders_and_missing_keys(moe_config, tmp_path):
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
        orig.use_torch_mm = True
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].experts = GroupedExpertsNF4(orig)
    experts = model.layers[0].experts

    (tmp_path / "empty").mkdir()
    from safetensors.torch import save_file

    save_file({"unrelated": torch.zeros(1)}, str(tmp_path / "empty" / "model.safetensors"))
    with pytest.raises(KeyError, match="per-expert HF layout"):
        load_nf4_experts_from_hf_checkpoint(model, str(tmp_path / "empty"), device=torch.device("cpu"))

    experts.register_parameter(
        "down_projs_nf4", nn.Parameter(torch.zeros_like(experts.down_projs_nf4, device="cpu"), requires_grad=False)
    )
    with pytest.raises(RuntimeError, match="materialized"):
        assert_nf4_placeholders_on_meta(model)
