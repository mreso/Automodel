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

"""NF4 (4-bit NormalFloat, the QLoRA data type) pack/unpack utilities for MoE expert weights.

The packed layout is the bitsandbytes 4-bit layout: two NF4 codes per ``uint8`` byte (the even
element in the high nibble) and one ``float32`` absmax per ``NF4_BLOCK_SIZE`` contiguous elements
along the last dim, so ``0.5625`` bytes per parameter. ``quantize_nf4`` / ``dequantize_nf4`` run
the bitsandbytes CUDA kernels when they are usable and otherwise a pure-torch implementation of
the same format (CPU tensors, or a bitsandbytes build without a matching CUDA library).
``NF4GroupedMM`` provides a grouped GEMM over packed weights that re-dequantizes in backward
instead of saving the dequantized tensor, so frozen expert weights stay packed at steady state
during LoRA training.
"""

import logging

import torch
from torch.autograd.function import FunctionCtx

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

HAS_BNB, bitsandbytes = safe_import("bitsandbytes")

NF4_BLOCK_SIZE = 64

# NF4 levels (Dettmers et al., QLoRA, Appendix E), ascending; the same table bitsandbytes uses.
NF4_LEVELS = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
_NF4_TABLE = torch.tensor(NF4_LEVELS, dtype=torch.float32)
_NF4_MIDPOINTS = (_NF4_TABLE[1:] + _NF4_TABLE[:-1]) / 2
# Both NF4 values of a byte, high nibble first, so one gather over the byte table unpacks a tensor.
_NF4_BYTE_TABLE = torch.stack(
    [
        _NF4_TABLE[(torch.arange(256) >> 4) & 0x0F],
        _NF4_TABLE[torch.arange(256) & 0x0F],
    ],
    dim=-1,
)  # [256, 2]

# Whether the bitsandbytes CUDA kernels work in this process; resolved on first use.
_BNB_KERNELS = {"usable": None}


def _bnb_usable(device: torch.device) -> bool:
    """Whether ``quantize_nf4``/``dequantize_nf4`` should use the bitsandbytes kernels on ``device``."""
    if device.type != "cuda" or not HAS_BNB:
        return False
    if _BNB_KERNELS["usable"] is None:
        try:
            probe = torch.zeros(2 * NF4_BLOCK_SIZE, dtype=torch.bfloat16, device=device)
            packed, state = bitsandbytes.functional.quantize_4bit(
                probe, blocksize=NF4_BLOCK_SIZE, quant_type="nf4", compress_statistics=False
            )
            bitsandbytes.functional.dequantize_4bit(
                packed, quant_state=state, blocksize=NF4_BLOCK_SIZE, quant_type="nf4"
            )
            _BNB_KERNELS["usable"] = True
            logger.info("NF4 expert quantization uses the bitsandbytes %s kernels", bitsandbytes.__version__)
        except Exception as exc:  # pragma: no cover - depends on the installed bitsandbytes build
            logger.warning("bitsandbytes NF4 kernels unavailable (%s); using the torch NF4 implementation", exc)
            _BNB_KERNELS["usable"] = False
    return _BNB_KERNELS["usable"]


def _check_last_dim(k: int) -> None:
    if k % NF4_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {k} must be divisible by {NF4_BLOCK_SIZE} for NF4 block quantization")


def _quantize_nf4_torch(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference NF4 quantization in plain torch (same layout as the bitsandbytes kernels)."""
    k = weight.shape[-1]
    blocks = weight.float().reshape(-1, NF4_BLOCK_SIZE)
    absmax = blocks.abs().amax(dim=-1)
    # bitsandbytes scales by the reciprocal of the block absmax; an all-zero block keeps its codes at 0.
    inv = torch.where(absmax > 0, 1.0 / absmax, torch.ones_like(absmax))
    codes = torch.bucketize(blocks * inv.unsqueeze(-1), _NF4_MIDPOINTS.to(weight.device)).to(torch.uint8).reshape(-1)
    packed = (codes[0::2] << 4) | codes[1::2]
    return packed.reshape(*weight.shape[:-1], k // 2), absmax.reshape(*weight.shape[:-1], k // NF4_BLOCK_SIZE)


def _dequantize_nf4_torch(codes: torch.Tensor, absmax: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reference NF4 dequantization in plain torch."""
    pairs = _NF4_BYTE_TABLE.to(codes.device)[codes.long()]  # [..., K // 2, 2]
    values = pairs.flatten(-2)  # [..., K]
    blocked = values.view(*values.shape[:-1], absmax.shape[-1], NF4_BLOCK_SIZE)
    return (blocked * absmax.to(torch.float32).unsqueeze(-1)).flatten(-2).to(dtype)


def quantize_nf4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dim to packed NF4 codes plus per-block float32 absmax.

    Args:
        weight: Floating-point tensor of shape ``[..., K]`` with ``K`` divisible by ``NF4_BLOCK_SIZE``.

    Returns:
        Tuple of (``uint8`` packed codes ``[..., K // 2]``, ``float32`` absmax ``[..., K // NF4_BLOCK_SIZE]``).
    """
    k = weight.shape[-1]
    _check_last_dim(k)
    if _bnb_usable(weight.device):
        flat = weight.reshape(-1).contiguous()
        if flat.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            flat = flat.to(torch.bfloat16)
        packed, state = bitsandbytes.functional.quantize_4bit(
            flat, blocksize=NF4_BLOCK_SIZE, quant_type="nf4", compress_statistics=False
        )
        codes = packed.view(torch.uint8).reshape(*weight.shape[:-1], k // 2)
        absmax = state.absmax.to(torch.float32).reshape(*weight.shape[:-1], k // NF4_BLOCK_SIZE)
        return codes.contiguous(), absmax.contiguous()
    return _quantize_nf4_torch(weight)


def dequantize_nf4(codes: torch.Tensor, absmax: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack NF4 codes and apply the per-block absmax.

    Args:
        codes: ``uint8`` tensor of shape ``[..., K // 2]`` holding two NF4 codes per byte.
        absmax: ``float32`` tensor of shape ``[..., K // NF4_BLOCK_SIZE]``.
        dtype: Output dtype.

    Returns:
        Dequantized tensor of shape ``[..., K]`` in ``dtype``.
    """
    shape = (*codes.shape[:-1], codes.shape[-1] * 2)
    if dtype in (torch.float16, torch.bfloat16, torch.float32) and _bnb_usable(codes.device):
        out = torch.empty(shape, dtype=dtype, device=codes.device)
        bitsandbytes.functional.dequantize_4bit(
            codes.reshape(-1, 1),
            absmax=absmax.reshape(-1).to(torch.float32),
            out=out,
            blocksize=NF4_BLOCK_SIZE,
            quant_type="nf4",
        )
        return out
    return _dequantize_nf4_torch(codes, absmax, dtype)


class NF4GroupedMM(torch.autograd.Function):
    """Grouped GEMM over NF4-packed frozen weights with dequantization on the fly.

    Saves only the packed codes and absmax for backward and re-dequantizes there, so the
    floating-point weight tensor is a transient in both passes instead of being kept alive
    by autograd. Weights are stored as ``[E, N, K]`` packed along ``K``, the natural
    dequantization output and the operand the backward GEMM needs directly
    (``grad_x = grad_out @ W``). The forward needs ``[E, K, N]``, which ``torch._grouped_mm``
    consumes as a transposed view (cuBLAS transB) without a contiguous copy.

    No weight gradient is produced: the base weights are frozen under LoRA.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, codes: torch.Tensor, absmax: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Multiply activations grouped by expert by packed frozen weights.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            codes: ``uint8`` tensor of shape [local_experts, out_dim, in_dim // 2].
            absmax: ``float32`` tensor of shape [local_experts, out_dim, in_dim // NF4_BLOCK_SIZE].
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.
                All tensors must be on the same device.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype.
        """
        w_t = dequantize_nf4(codes, absmax, x.dtype)  # [E, N, K]
        out = torch._grouped_mm(x, w_t.transpose(-2, -1), offs=offs)
        ctx.save_for_backward(codes, absmax, offs)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Backpropagate through activations while keeping the packed base frozen.

        Args:
            grad_out: Tensor of shape [tokens, out_dim], in the forward output dtype.

        Returns:
            Activation gradient of shape [tokens, in_dim], followed by None for the codes,
            absmax and offsets.
        """
        codes, absmax, offs = ctx.saved_tensors
        w_t = dequantize_nf4(codes, absmax, grad_out.dtype)  # [E, N, K] == W^T
        grad_x = torch._grouped_mm(grad_out.contiguous(), w_t, offs=offs)
        return grad_x, None, None, None
