# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Quantized-resident expert storage and model application for frozen MoE experts (MXFP4 and NF4).

``GroupedExpertsMXFP4`` keeps the frozen routed-expert base weights packed as
fp4-e2m1 + e8m0 block scales (the DeepSeek V4 Flash checkpoint format) and
dequantizes on the fly inside the grouped GEMM, instead of holding them in
bf16. This is the storage win for LoRA / frozen-base training of large MoE
models, where the routed experts dominate parameter memory.

``MXFP4ExpertStorageMixin`` shares packed storage and base GEMMs between the
frozen and LoRA experts, with Torch or DeepEP token dispatch.

``NF4ExpertStorageMixin`` / ``GroupedExpertsNF4`` do the same with NF4 (QLoRA) storage for
models whose checkpoints are floating point: the packed placeholders are streamed from the
per-expert HF checkpoint by ``load_nf4_experts_from_hf_checkpoint`` after sharding.
"""

import json
import logging
import os
import re
import time

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.components.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    MXFP4GroupedMM,
    dequantize_mxfp4,
    quantize_mxfp4,
)
from nemo_automodel.components.quantization.nf4 import (
    NF4_BLOCK_SIZE,
    NF4GroupedMM,
    dequantize_nf4,
    quantize_nf4,
)
from nemo_automodel.shared.parameter_names import SKIP_CHECKPOINT_LOAD_ATTR

logger = logging.getLogger(__name__)


def _to_local(t):
    """Return the local shard of a DTensor, or the tensor unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


class MXFP4ExpertStorageMixin:
    """Packed-mxfp4 base-weight storage and grouped GEMM for routed experts.

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as packed fp4
    (int8, two e2m1 nibbles per byte) plus ``float8_e8m0fnu`` block scales, in
    checkpoint orientation ``[n_experts, out_dim, in_dim]`` so the block scales run
    along the contraction dim. Floating-point base parameters are dropped once packed.

    Meta weights become packed placeholders for direct checkpoint loading.
    Materialized weights are quantized immediately during construction.
    """

    _MXFP4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")

    def _validate_mxfp4_config(self) -> None:
        """Reject execution modes that the packed expert computation does not implement."""
        if self.config.apply_router_weight_after_down:
            raise NotImplementedError("MXFP4 experts do not support apply_router_weight_after_down=True.")
        # DeepEP experts always use native grouped MM; only plain experts can select the loop backend.
        if isinstance(self, GroupedExperts) and not self.use_torch_mm:
            raise NotImplementedError(
                "mxfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The per-expert loop has no packed variant; with DeepEP dispatch use "
                "backend.dispatcher='deepep' together with backend.experts='torch_mm'."
            )

    def _init_mxfp4_storage(self) -> None:
        """Replace floating-point bases with packed placeholders or quantized weights."""
        self._validate_mxfp4_config()
        if _to_local(self.gate_and_up_projs).is_meta:
            self._init_packed_placeholders()
        else:
            self._pack_base_weights()
        # The shared expert initializer must skip the removed floating-point bases.
        self._mxfp4_resident = True

    @torch.no_grad()
    def _init_packed_placeholders(self) -> None:
        """Register packed meta parameters in checkpoint orientation [experts, out, in]."""
        cfg = self.config
        block = MXFP4_BLOCK_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % block == 0 and moe_inter % block == 0, (
            f"expert dims must be divisible by {block} for mxfp4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim.
        shapes = {
            "gate_and_up_projs": ((e, up_proj_dim, expert_dim // 2), (e, up_proj_dim, expert_dim // block)),
            "down_projs": ((e, expert_dim, moe_inter // 2), (e, expert_dim, moe_inter // block)),
        }
        for name, (packed_shape, scale_shape) in shapes.items():
            packed = torch.empty(packed_shape, dtype=torch.int8, device="meta")
            scales = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device="meta")
            self._register_packed_base_weight(name, packed, scales)

    def _register_packed_base_weight(self, name: str, packed: torch.Tensor, scales: torch.Tensor) -> None:
        """Replace one floating-point base projection with frozen packed parameters.

        Args:
            name: Base projection parameter name.
            packed: Int8 tensor of shape [experts, out_dim, in_dim // 2].
            scales: E8M0 tensor of shape [experts, out_dim, in_dim // 32].
                DTensors must retain the base parameter's mesh and expert-axis placement.
        """
        del self._parameters[name]
        self.register_parameter(name + "_packed", nn.Parameter(packed, requires_grad=False))
        self.register_parameter(name + "_scales", nn.Parameter(scales, requires_grad=False))

    @torch.no_grad()
    def _pack_base_weights(self) -> None:
        """Quantize materialized base projections and release their floating-point storage."""
        for name in self._MXFP4_BASE_NAMES:
            param = getattr(self, name)
            # Compute layout [experts, in, out] -> checkpoint layout [experts, out, in].
            packed, scales = quantize_mxfp4(_to_local(param).transpose(-2, -1).contiguous())
            if isinstance(param, DTensor):
                packed = DTensor.from_local(packed, param.device_mesh, param.placements)
                scales = DTensor.from_local(scales, param.device_mesh, param.placements)
            self._register_packed_base_weight(name, packed, scales)

    def _mxfp4_base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Multiply routed activations by a frozen packed base projection.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            name: Base projection parameter name.
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype and device.
        """
        packed = _to_local(getattr(self, name + "_packed"))
        scales = _to_local(getattr(self, name + "_scales"))
        return MXFP4GroupedMM.apply(x, packed, scales, offs)

    def _mxfp4_dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed = _to_local(getattr(self, name + "_packed"))[0]
        scales = _to_local(getattr(self, name + "_scales"))[0]
        return dequantize_mxfp4(packed, scales, dtype).transpose(-2, -1)


class GroupedExpertsMXFP4(MXFP4ExpertStorageMixin, GroupedExperts):
    """Frozen routed experts with mxfp4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g.
    LoRA training that targets only attention). Forward mirrors
    ``GroupedExperts._forward_grouped_mm`` but reads the packed base weights.
    """

    def __init__(self, orig_module: GroupedExperts) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(orig_module.config, backend=None)
        self.use_torch_mm = orig_module.use_torch_mm
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._MXFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute packed experts with the tensor and EP contract of GroupedExperts.forward."""
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs_packed, DTensor):
            ep_mesh = self.gate_and_up_projs_packed.device_mesh
            assert ep_mesh is not None and ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_num_tokens = x.size(0)
            x, weights, indices, token_mask, gathered_lens = self._gather_ep_inputs(
                x, token_mask, weights, indices, ep_group=ep_group, ep_size=ep_size
            )

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts

        y = self._forward_grouped_mm_mxfp4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)

            # Reduce and narrow to the original per-rank token boundaries.
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_grouped_mm_mxfp4(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        n_local_experts: int,
        experts_start_idx: int,
    ) -> torch.Tensor:
        """Compute the frozen local experts' contribution before the EP reduction.

        Args:
            x: Tensor of shape [tokens, hidden], gathered across the EP group.
            token_mask: Boolean tensor of shape [tokens] selecting valid tokens.
            weights: Tensor of shape [tokens, top_k] with differentiable routing probabilities.
            indices: Integer tensor of shape [tokens, top_k] with global expert IDs.
            n_local_experts: Number of experts on this rank.
            experts_start_idx: First global expert ID on this rank.

        Returns:
            FP32 tensor of shape [tokens, hidden], before the EP reduction.
        """
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts, experts_start_idx
        )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            output1 = self._mxfp4_base_mm(permuted_x, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation_grouped(output1, permuted_probs)

            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPMXFP4(MXFP4ExpertStorageMixin, GroupedExpertsDeepEP):
    """Frozen routed experts with mxfp4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen
    (e.g. LoRA on attention only). The DeepEP fused all-to-all token dispatch is reused
    unchanged — mxfp4 only changes the two post-dispatch grouped GEMMs, which read the
    packed base weights via ``MXFP4GroupedMM`` instead of bf16 ``torch._grouped_mm``.

    The DeepEP parent always uses native grouped MM.
    """

    def __init__(self, orig_module: GroupedExpertsDeepEP) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(
                orig_module.config,
                backend=None,
                dispatcher_backend=orig_module.dispatcher_backend,
                dispatcher_num_sms=orig_module.dispatcher_num_sms,
                dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
                dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
            )
        self.dispatcher_hybridep_permute_fusion = orig_module.dispatcher_hybridep_permute_fusion
        self.dispatcher_hybridep_compact_routing = orig_module.dispatcher_hybridep_compact_routing
        self.dispatcher_hybridep_num_sms_preprocessing = orig_module.dispatcher_hybridep_num_sms_preprocessing
        self.dispatcher_hybridep_num_blocks_permute = orig_module.dispatcher_hybridep_num_blocks_permute
        self.dispatcher_hybridep_num_blocks_unpermute = orig_module.dispatcher_hybridep_num_blocks_unpermute
        # Sync-free dispatch knobs (BackendConfig.dispatcher_capacity_factor / dispatcher_equal_token_counts) travel
        # with the other HybridEP settings: the shared dispatcher is built from the first (wrapped) layer.
        self.dispatcher_capacity_factor = orig_module.dispatcher_capacity_factor
        self.dispatcher_equal_token_counts = orig_module.dispatcher_equal_token_counts
        self.dispatch_rows_never_empty = orig_module.dispatch_rows_never_empty
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._MXFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over mxfp4 base weights with DeepEP dispatch.

        Preserves the tensor and EP contract of ``GroupedExpertsDeepEP.forward``, replacing the two base
        ``torch._grouped_mm`` calls with ``MXFP4GroupedMM`` over the packed weights.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        # dispatch_rows_never_empty (static routing or capacity mode): skip the count_nonzero host read, as
        # GroupedExpertsDeepEP.forward does.
        if self.dispatch_rows_never_empty or torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            output1 = self._mxfp4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y


def apply_mxfp4_to_moe_experts(model: nn.Module) -> nn.Module:
    """Apply MXFP4-resident storage to the model's common routed-expert modules.

    Call this after LoRA injection and before distributed sharding. LoRA-targeted
    experts that are already MXFP4-resident are preserved; remaining plain
    ``GroupedExperts`` and ``GroupedExpertsDeepEP`` modules are replaced in place.

    Meta experts receive packed placeholders for direct checkpoint loading, and
    their model state-dict adapter must opt into the MXFP4 expert storage format.
    Already materialized expert weights are quantized without changing the adapter's
    checkpoint loading mode. Model-specific adapters own checkpoint layouts.
    """
    model_parts = list(model.parts) if hasattr(model, "parts") else [model]

    unsupported = [
        name
        for model_part in model_parts
        for name, module in model_part.named_modules()
        if isinstance(module, GroupedExpertsTE)
    ]
    if unsupported:
        raise NotImplementedError(
            "MXFP4-resident expert weights do not support Transformer Engine expert modules; "
            "use backend.experts='torch_mm'."
        )

    for model_part in model_parts:
        needs_checkpoint = any(
            param.is_meta
            for module in model_part.modules()
            if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP))
            for param in module.parameters(recurse=False)
        )
        if needs_checkpoint:
            adapter = getattr(model_part, "state_dict_adapter", None)
            set_storage_format = getattr(adapter, "set_expert_storage_format", None)
            if not callable(set_storage_format):
                raise NotImplementedError(
                    "Loading an MXFP4 expert checkpoint requires the model state-dict adapter to implement "
                    "set_expert_storage_format()."
                )
            set_storage_format("mxfp4")

    frozen_conversions = {
        GroupedExperts: GroupedExpertsMXFP4,
        GroupedExpertsDeepEP: GroupedExpertsDeepEPMXFP4,
    }
    num_converted = 0
    for model_part in model_parts:
        for name, module in list(model_part.named_modules()):
            if isinstance(module, MXFP4ExpertStorageMixin):
                continue
            new_cls = frozen_conversions.get(type(module))
            if new_cls is None:
                continue
            new_module = new_cls(module)
            parent_name, _, child_name = name.rpartition(".")
            parent = model_part.get_submodule(parent_name) if parent_name else model_part
            setattr(parent, child_name, new_module)
            num_converted += 1

    logger.info("Applied MXFP4-resident storage to %d frozen expert module(s)", num_converted)
    return model


class NF4ExpertStorageMixin:
    """Packed-NF4 base-weight storage and grouped GEMM for routed experts (QLoRA for MoE experts).

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as packed NF4 codes
    (``uint8``, two 4-bit codes per byte) plus one ``float32`` absmax per
    ``NF4_BLOCK_SIZE`` elements, in checkpoint orientation ``[n_experts, out_dim, in_dim]``
    so the blocks run along the contraction dim: 0.5625 bytes per parameter. Floating-point
    base parameters are dropped once packed.

    Meta weights become packed placeholders that ``load_nf4_experts_from_hf_checkpoint``
    streams from a floating-point per-expert HF checkpoint after sharding, one expert at a
    time; the DCP base load leaves them alone. Materialized weights are quantized
    immediately during construction.
    """

    _NF4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")

    @staticmethod
    def nf4_parameter_names(name: str) -> tuple[str, str]:
        """Names of the packed codes and block absmax parameters replacing base projection ``name``."""
        return name + "_nf4", name + "_absmax"

    def _validate_nf4_config(self) -> None:
        """Reject execution modes that the packed expert computation does not implement."""
        if self.config.apply_router_weight_after_down:
            raise NotImplementedError("NF4 experts do not support apply_router_weight_after_down=True.")
        # DeepEP experts always use native grouped MM; only plain experts can select the loop backend.
        if isinstance(self, GroupedExperts) and not self.use_torch_mm:
            raise NotImplementedError(
                "nf4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The per-expert loop has no packed variant; with DeepEP dispatch use "
                "backend.dispatcher='deepep' together with backend.experts='torch_mm'."
            )

    def _init_nf4_storage(self) -> None:
        """Replace floating-point bases with packed placeholders or quantized weights."""
        self._validate_nf4_config()
        if _to_local(self.gate_and_up_projs).is_meta:
            self._init_nf4_placeholders()
        else:
            self._pack_base_weights_nf4()
        # The shared expert initializer must skip the removed floating-point bases.
        self._nf4_resident = True

    @torch.no_grad()
    def _init_nf4_placeholders(self) -> None:
        """Register packed meta parameters in checkpoint orientation [experts, out, in]."""
        cfg = self.config
        block = NF4_BLOCK_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % block == 0 and moe_inter % block == 0, (
            f"expert dims must be divisible by {block} for nf4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim.
        shapes = {
            "gate_and_up_projs": ((e, up_proj_dim, expert_dim // 2), (e, up_proj_dim, expert_dim // block)),
            "down_projs": ((e, expert_dim, moe_inter // 2), (e, expert_dim, moe_inter // block)),
        }
        for name, (codes_shape, absmax_shape) in shapes.items():
            codes = torch.empty(codes_shape, dtype=torch.uint8, device="meta")
            absmax = torch.empty(absmax_shape, dtype=torch.float32, device="meta")
            self._register_nf4_base_weight(name, codes, absmax)
        # The placeholders are filled by load_nf4_experts_from_hf_checkpoint, not by the base checkpoint load:
        # keep them on meta through materialization and out of the DCP load destinations.
        setattr(
            self,
            SKIP_CHECKPOINT_LOAD_ATTR,
            {param for name in self._NF4_BASE_NAMES for param in self.nf4_parameter_names(name)},
        )

    def _register_nf4_base_weight(self, name: str, codes: torch.Tensor, absmax: torch.Tensor) -> None:
        """Replace one floating-point base projection with frozen packed parameters.

        Args:
            name: Base projection parameter name.
            codes: ``uint8`` tensor of shape [experts, out_dim, in_dim // 2].
            absmax: ``float32`` tensor of shape [experts, out_dim, in_dim // NF4_BLOCK_SIZE].
                DTensors must retain the base parameter's mesh and expert-axis placement.
        """
        codes_name, absmax_name = self.nf4_parameter_names(name)
        del self._parameters[name]
        self.register_parameter(codes_name, nn.Parameter(codes, requires_grad=False))
        self.register_parameter(absmax_name, nn.Parameter(absmax, requires_grad=False))

    @torch.no_grad()
    def _pack_base_weights_nf4(self) -> None:
        """Quantize materialized base projections and release their floating-point storage."""
        for name in self._NF4_BASE_NAMES:
            param = getattr(self, name)
            # Compute layout [experts, in, out] -> checkpoint layout [experts, out, in].
            codes, absmax = quantize_nf4(_to_local(param).transpose(-2, -1).contiguous())
            if isinstance(param, DTensor):
                codes = DTensor.from_local(codes, param.device_mesh, param.placements)
                absmax = DTensor.from_local(absmax, param.device_mesh, param.placements)
            self._register_nf4_base_weight(name, codes, absmax)

    def _nf4_base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Multiply routed activations by a frozen packed base projection.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            name: Base projection parameter name.
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype and device.
        """
        codes_name, absmax_name = self.nf4_parameter_names(name)
        codes = _to_local(getattr(self, codes_name))
        absmax = _to_local(getattr(self, absmax_name))
        return NF4GroupedMM.apply(x, codes, absmax, offs)

    def _nf4_dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        codes_name, absmax_name = self.nf4_parameter_names(name)
        codes = _to_local(getattr(self, codes_name))[0]
        absmax = _to_local(getattr(self, absmax_name))[0]
        return dequantize_nf4(codes, absmax, dtype).transpose(-2, -1)


class GroupedExpertsNF4(NF4ExpertStorageMixin, GroupedExperts):
    """Frozen routed experts with NF4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g.
    LoRA training that targets only attention). Forward mirrors
    ``GroupedExperts._forward_grouped_mm`` but reads the packed base weights.
    """

    def __init__(self, orig_module: GroupedExperts) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(orig_module.config, backend=None)
        self.use_torch_mm = orig_module.use_torch_mm
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._NF4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_nf4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute packed experts with the tensor and EP contract of GroupedExperts.forward."""
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs_nf4, DTensor):
            ep_mesh = self.gate_and_up_projs_nf4.device_mesh
            assert ep_mesh is not None and ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_num_tokens = x.size(0)
            x, weights, indices, token_mask, gathered_lens = self._gather_ep_inputs(
                x, token_mask, weights, indices, ep_group=ep_group, ep_size=ep_size
            )

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts

        y = self._forward_grouped_mm_nf4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)

            # Reduce and narrow to the original per-rank token boundaries.
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_grouped_mm_nf4(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        n_local_experts: int,
        experts_start_idx: int,
    ) -> torch.Tensor:
        """Compute the frozen local experts' contribution before the EP reduction.

        Args:
            x: Tensor of shape [tokens, hidden], gathered across the EP group.
            token_mask: Boolean tensor of shape [tokens] selecting valid tokens.
            weights: Tensor of shape [tokens, top_k] with differentiable routing probabilities.
            indices: Integer tensor of shape [tokens, top_k] with global expert IDs.
            n_local_experts: Number of experts on this rank.
            experts_start_idx: First global expert ID on this rank.

        Returns:
            FP32 tensor of shape [tokens, hidden], before the EP reduction.
        """
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts, experts_start_idx
        )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            output1 = self._nf4_base_mm(permuted_x, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation_grouped(output1, permuted_probs)

            output2 = self._nf4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._nf4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._nf4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPNF4(NF4ExpertStorageMixin, GroupedExpertsDeepEP):
    """Frozen routed experts with NF4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen
    (e.g. LoRA on attention only). The DeepEP fused all-to-all token dispatch is reused
    unchanged — NF4 only changes the two post-dispatch grouped GEMMs, which read the
    packed base weights via ``NF4GroupedMM`` instead of floating-point ``torch._grouped_mm``.

    The DeepEP parent always uses native grouped MM.
    """

    def __init__(self, orig_module: GroupedExpertsDeepEP) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(
                orig_module.config,
                backend=None,
                dispatcher_backend=orig_module.dispatcher_backend,
                dispatcher_num_sms=orig_module.dispatcher_num_sms,
                dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
                dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
            )
        self.dispatcher_hybridep_permute_fusion = orig_module.dispatcher_hybridep_permute_fusion
        self.dispatcher_hybridep_compact_routing = orig_module.dispatcher_hybridep_compact_routing
        self.dispatcher_hybridep_num_sms_preprocessing = orig_module.dispatcher_hybridep_num_sms_preprocessing
        self.dispatcher_hybridep_num_blocks_permute = orig_module.dispatcher_hybridep_num_blocks_permute
        self.dispatcher_hybridep_num_blocks_unpermute = orig_module.dispatcher_hybridep_num_blocks_unpermute
        # Sync-free dispatch knobs (BackendConfig.dispatcher_capacity_factor / dispatcher_equal_token_counts) travel
        # with the other HybridEP settings: the shared dispatcher is built from the first (wrapped) layer.
        self.dispatcher_capacity_factor = orig_module.dispatcher_capacity_factor
        self.dispatcher_equal_token_counts = orig_module.dispatcher_equal_token_counts
        self.dispatch_rows_never_empty = orig_module.dispatch_rows_never_empty
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._NF4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_nf4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over NF4 base weights with DeepEP dispatch.

        Preserves the tensor and EP contract of ``GroupedExpertsDeepEP.forward``, replacing the two base
        ``torch._grouped_mm`` calls with ``NF4GroupedMM`` over the packed weights.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        # dispatch_rows_never_empty (static routing or capacity mode): skip the count_nonzero host read, as
        # GroupedExpertsDeepEP.forward does.
        if self.dispatch_rows_never_empty or torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            output1 = self._nf4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self._nf4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._nf4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._nf4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y


def apply_nf4_to_moe_experts(model: nn.Module) -> nn.Module:
    """Apply NF4-resident storage to the model's common routed-expert modules.

    Call this after LoRA injection and before distributed sharding. LoRA-targeted
    experts that are already NF4-resident are preserved; remaining plain
    ``GroupedExperts`` and ``GroupedExpertsDeepEP`` modules are replaced in place.

    Meta experts receive packed placeholders that ``load_nf4_experts_from_hf_checkpoint``
    streams from the floating-point HF checkpoint after the base load; already
    materialized expert weights are quantized immediately.
    """
    model_parts = list(model.parts) if hasattr(model, "parts") else [model]

    unsupported = [
        name
        for model_part in model_parts
        for name, module in model_part.named_modules()
        if isinstance(module, GroupedExpertsTE)
    ]
    if unsupported:
        raise NotImplementedError(
            "NF4-resident expert weights do not support Transformer Engine expert modules; use backend.experts='torch_mm'."
        )

    frozen_conversions = {
        GroupedExperts: GroupedExpertsNF4,
        GroupedExpertsDeepEP: GroupedExpertsDeepEPNF4,
    }
    num_converted = 0
    for model_part in model_parts:
        for name, module in list(model_part.named_modules()):
            if isinstance(module, NF4ExpertStorageMixin):
                continue
            new_cls = frozen_conversions.get(type(module))
            if new_cls is None:
                continue
            new_module = new_cls(module)
            parent_name, _, child_name = name.rpartition(".")
            parent = model_part.get_submodule(parent_name) if parent_name else model_part
            setattr(parent, child_name, new_module)
            num_converted += 1

    logger.info("Applied NF4-resident storage to %d frozen expert module(s)", num_converted)
    return model


def _nf4_pending_modules(model: nn.Module):
    """Yield (name, module) of NF4-resident experts whose placeholders still await the checkpoint stream."""
    for name, module in model.named_modules():
        if getattr(module, "_nf4_resident", False) and getattr(module, SKIP_CHECKPOINT_LOAD_ATTR, None):
            yield name, module


def nf4_placeholders_pending(model: nn.Module) -> bool:
    """Whether any NF4-resident expert module still holds unfilled packed placeholders."""
    return next(_nf4_pending_modules(model), None) is not None


def assert_nf4_placeholders_on_meta(model: nn.Module) -> None:
    """Fail fast (before the base load) if pending NF4 placeholders were materialized."""
    for name, module in _nf4_pending_modules(model):
        for param_name in getattr(module, SKIP_CHECKPOINT_LOAD_ATTR):
            if not getattr(module, param_name).is_meta:
                raise RuntimeError(
                    f"{name}.{param_name}: NF4 placeholder was materialized before the checkpoint stream"
                )


class _SafetensorsIndex:
    """Lazy per-tensor reader over a sharded HF safetensors checkpoint (one open handle per shard file)."""

    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open

            with safe_open(os.path.join(checkpoint_dir, "model.safetensors"), framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles = {}

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        fname = self.weight_map[key]
        handle = self._handles.get(fname)
        if handle is None:
            handle = safe_open(os.path.join(self.dir, fname), framework="pt", device="cpu")
            self._handles[fname] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


@torch.no_grad()
def load_nf4_experts_from_hf_checkpoint(
    model: nn.Module, checkpoint_dir: str, *, device: torch.device | None = None
) -> dict:
    """Stream the frozen experts of every NF4-resident module from a floating-point per-expert HF checkpoint.

    Fills the packed placeholders created on meta (see ``NF4ExpertStorageMixin``) one expert at a time, so
    the peak expert memory is the NF4 storage plus one expert in the checkpoint dtype instead of the full
    floating-point experts. Each rank reads only its own expert-parallel shard. The per-expert HF layout
    ``{prefix}layers.{L}.{experts}.{E}.{gate,up,down}_proj.weight`` ([out, in] matrices) is taken from the
    model's state-dict adapter (``_hf_prefix`` / ``_expert_path_segment``), defaulting to the Qwen3-MoE style
    ``model.layers.{L}.mlp.experts.{E}``.

    Args:
        model: Model (part) whose NF4-resident experts hold unfilled placeholders.
        checkpoint_dir: Local directory with the HF safetensors checkpoint.
        device: Device of the packed storage; defaults to the current CUDA device, else CPU.

    Returns:
        Per-rank statistics (modules, experts, bytes read, packed bytes, seconds).
    """
    adapter = getattr(model, "state_dict_adapter", None)
    hf_prefix = getattr(adapter, "_hf_prefix", "model.")
    expert_segment = getattr(adapter, "_expert_path_segment", "mlp.experts")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    t0 = time.monotonic()
    index = _SafetensorsIndex(checkpoint_dir)
    stats = {"modules": 0, "experts": 0, "bytes_read": 0, "nf4_bytes": 0}
    for name, module in list(_nf4_pending_modules(model)):
        pending = set(getattr(module, SKIP_CHECKPOINT_LOAD_ATTR))
        expected = {param for base in module._NF4_BASE_NAMES for param in module.nf4_parameter_names(base)}
        if pending != expected:
            raise RuntimeError(f"{name}: unexpected NF4 placeholder set {sorted(pending)}")
        match = re.search(r"layers\.(\d+)\.", name + ".")
        if match is None:
            raise ValueError(f"{name}: cannot find the decoder layer index needed for the HF expert keys")
        layer = int(match.group(1))

        placeholders = {
            base: tuple(getattr(module, p) for p in module.nf4_parameter_names(base)) for base in module._NF4_BASE_NAMES
        }
        for base, (codes_p, absmax_p) in placeholders.items():
            if not (codes_p.is_meta and absmax_p.is_meta):
                raise RuntimeError(f"{name}.{base}: NF4 placeholders were materialized before the checkpoint stream")
        reference = placeholders["gate_and_up_projs"][0]
        if isinstance(reference, DTensor):
            mesh, placements = reference.device_mesh, reference.placements
            assert mesh.ndim == 1 and isinstance(placements[0], Shard) and placements[0].dim == 0, placements
            ep_size, ep_rank = mesh.size(), mesh.get_local_rank()
        else:
            mesh = placements = None
            ep_size, ep_rank = 1, 0
        n_local = module.n_routed_experts // ep_size
        first = ep_rank * n_local

        local = {}
        for base, (codes_p, absmax_p) in placeholders.items():
            local[base] = (
                torch.empty((n_local, *codes_p.shape[1:]), dtype=torch.uint8, device=device),
                torch.empty((n_local, *absmax_p.shape[1:]), dtype=torch.float32, device=device),
            )
        projections = ("gate_proj", "up_proj", "down_proj") if module.is_gated else ("up_proj", "down_proj")
        key_base = f"{hf_prefix}layers.{layer}.{expert_segment}"
        for i in range(n_local):
            keys = {proj: f"{key_base}.{first + i}.{proj}.weight" for proj in projections}
            missing = [key for key in keys.values() if key not in index]
            if missing:
                raise KeyError(f"{missing[0]} not in {checkpoint_dir}: NF4 streaming needs the per-expert HF layout")
            tensors = {proj: index.get(key).to(device=device) for proj, key in keys.items()}
            stats["bytes_read"] += sum(t.numel() * t.element_size() for t in tensors.values())
            gate_and_up = (
                torch.cat([tensors["gate_proj"], tensors["up_proj"]], dim=0) if module.is_gated else tensors["up_proj"]
            )
            for base, weight in (("gate_and_up_projs", gate_and_up), ("down_projs", tensors["down_proj"])):
                codes, absmax = local[base]
                if tuple(weight.shape) != (codes.shape[1], codes.shape[2] * 2):
                    raise ValueError(
                        f"{name}.{base}: checkpoint expert {first + i} has shape {tuple(weight.shape)}, "
                        f"expected {(codes.shape[1], codes.shape[2] * 2)}"
                    )
                packed, scales = quantize_nf4(weight)
                codes[i].copy_(packed)
                absmax[i].copy_(scales)
            del tensors, gate_and_up

        for base, (codes, absmax) in local.items():
            if mesh is not None:
                codes = DTensor.from_local(codes, mesh, placements, run_check=False)
                absmax = DTensor.from_local(absmax, mesh, placements, run_check=False)
            codes_name, absmax_name = module.nf4_parameter_names(base)
            module.register_parameter(codes_name, nn.Parameter(codes, requires_grad=False))
            module.register_parameter(absmax_name, nn.Parameter(absmax, requires_grad=False))
            stats["nf4_bytes"] += _to_local(codes).numel() + _to_local(absmax).numel() * 4
        delattr(module, SKIP_CHECKPOINT_LOAD_ATTR)
        stats["modules"] += 1
        stats["experts"] += n_local
    index.close()
    stats["seconds"] = round(time.monotonic() - t0, 1)
    if stats["modules"]:
        logger.info(
            "NF4 experts streamed from %s: %d modules, %d local experts, %.2f GiB read -> %.2f GiB packed in %.1f s",
            checkpoint_dir,
            stats["modules"],
            stats["experts"],
            stats["bytes_read"] / 2**30,
            stats["nf4_bytes"] / 2**30,
            stats["seconds"],
        )
    return stats
