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

from nemo_automodel.components.quantization import nf4 as nf4_module
from nemo_automodel.components.quantization.nf4 import (
    NF4_BLOCK_SIZE,
    NF4_LEVELS,
    NF4GroupedMM,
    _dequantize_nf4_torch,
    _quantize_nf4_torch,
    dequantize_nf4,
    quantize_nf4,
)

E, N, K = 3, 16, 128  # [experts, out, in]; K blockable by 64 and 16-byte aligned for the grouped GEMM


def _levels():
    return torch.tensor(NF4_LEVELS, dtype=torch.float32)


def _grouped_mm_available(device: torch.device) -> bool:
    x = torch.randn(4, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(2, K, N, dtype=torch.bfloat16, device=device)
    try:
        torch._grouped_mm(x, w, offs=torch.tensor([2, 4], dtype=torch.int32, device=device))
    except (RuntimeError, NotImplementedError):
        return False
    return True


def test_levels_are_sorted_with_unit_range():
    levels = _levels()
    assert torch.all(levels[1:] > levels[:-1])
    assert levels[0] == -1.0 and levels[-1] == 1.0 and 0.0 in NF4_LEVELS


def test_storage_layout_is_half_byte_per_param_plus_block_absmax():
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    codes, absmax = quantize_nf4(w)
    assert codes.dtype == torch.uint8 and codes.shape == (E, N, K // 2)
    assert absmax.dtype == torch.float32 and absmax.shape == (E, N, K // NF4_BLOCK_SIZE)
    assert codes.numel() + absmax.numel() * 4 == w.numel() * 0.5625


def test_last_dim_must_be_blockable():
    with pytest.raises(ValueError, match="divisible"):
        quantize_nf4(torch.randn(2, 8, 48))


def test_values_on_the_nf4_grid_round_trip_exactly():
    torch.manual_seed(0)
    levels = _levels()
    codes = torch.randint(0, 16, (E, N, K))
    codes[..., 0] = 15  # a +1.0 code in every block so the block absmax equals its scale exactly
    scales = torch.rand(E, N, K // NF4_BLOCK_SIZE) + 0.5
    w = (levels[codes].view(E, N, -1, NF4_BLOCK_SIZE) * scales.unsqueeze(-1)).reshape(E, N, K)
    q_codes, q_absmax = quantize_nf4(w)
    torch.testing.assert_close(q_absmax, scales)
    torch.testing.assert_close(dequantize_nf4(q_codes, q_absmax, torch.float32), w, rtol=0, atol=0)


def test_dequantize_error_is_bounded_by_half_the_level_gap():
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.float32)
    codes, absmax = quantize_nf4(w)
    err = (dequantize_nf4(codes, absmax, torch.float32) - w).abs().view(E, N, -1, NF4_BLOCK_SIZE)
    half_gap = (_levels()[1:] - _levels()[:-1]).max() / 2
    assert torch.all(err <= half_gap * absmax.unsqueeze(-1) + 1e-6)


def test_quantize_dequantize_idempotent():
    """A second quantize/dequantize round-trip must reproduce the first exactly."""
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    dq1 = dequantize_nf4(*quantize_nf4(w), torch.bfloat16)
    dq2 = dequantize_nf4(*quantize_nf4(dq1), torch.bfloat16)
    assert torch.equal(dq1, dq2)


def test_zero_block_decodes_to_zeros():
    w = torch.zeros(2, K, dtype=torch.bfloat16)
    w[1, NF4_BLOCK_SIZE:] = 1.5
    codes, absmax = quantize_nf4(w)
    dq = dequantize_nf4(codes, absmax, torch.bfloat16)
    assert torch.equal(dq[0], torch.zeros(K, dtype=torch.bfloat16))
    assert torch.equal(dq[1, :NF4_BLOCK_SIZE], torch.zeros(NF4_BLOCK_SIZE, dtype=torch.bfloat16))
    assert torch.equal(dq[1, NF4_BLOCK_SIZE:], torch.full((K - NF4_BLOCK_SIZE,), 1.5, dtype=torch.bfloat16))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_grouped_mm_dequantizes_and_backpropagates_to_the_input(device):
    device = torch.device(device)
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    torch.manual_seed(0)
    codes, absmax = quantize_nf4(torch.randn(E, N, K, dtype=torch.bfloat16, device=device))
    x = torch.randn(8, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    offs = torch.tensor([3, 3, 8], dtype=torch.int32, device=device)  # expert 1 receives no tokens
    out = NF4GroupedMM.apply(x, codes, absmax, offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    w_t = dequantize_nf4(codes, absmax, torch.bfloat16)  # [E, N, K]
    ref = torch._grouped_mm(x_ref, w_t.transpose(-2, -1), offs=offs)
    ref.backward(grad_out)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bitsandbytes_kernels_match_the_torch_reference():
    device = torch.device("cuda")
    if not nf4_module._bnb_usable(device):
        pytest.skip("bitsandbytes NF4 kernels are not usable here")
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    codes, absmax = quantize_nf4(w)  # bitsandbytes path
    ref_codes, ref_absmax = _quantize_nf4_torch(w)
    torch.testing.assert_close(absmax, ref_absmax)
    # Both round to the nearest level; only reciprocal-rounding ties may differ.
    assert (codes != ref_codes).float().mean().item() <= 1e-3
    bnb_dq = dequantize_nf4(codes, absmax, torch.float32)
    torch.testing.assert_close(bnb_dq, _dequantize_nf4_torch(codes, absmax, torch.float32), rtol=0, atol=1e-6)
