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

from nemo_automodel.components.checkpoint.checkpointing import (
    _drop_skip_load_entries,
    _require_no_skip_load_params,
    _skip_load_param_fqns,
    to_empty_parameters_only,
)
from nemo_automodel.shared.parameter_names import SKIP_CHECKPOINT_LOAD_ATTR


def test_to_empty_parameters_only_keeps_flagged_parameters_on_meta():
    module = nn.Linear(4, 2, device="meta")
    setattr(module, SKIP_CHECKPOINT_LOAD_ATTR, {"weight"})
    to_empty_parameters_only(module, device=torch.device("cpu"))
    assert module.bias.device.type == "cpu"
    assert module.weight.device.type == "meta"


def _model():
    outer = nn.Module()
    outer.experts = nn.Linear(4, 2, device="meta")
    outer.head = nn.Linear(2, 2)
    setattr(outer.experts, SKIP_CHECKPOINT_LOAD_ATTR, {"weight", "bias"})
    return outer


def test_skip_load_param_fqns_collects_flagged_parameters():
    assert _skip_load_param_fqns([_model()]) == {"experts.weight", "experts.bias"}
    assert _skip_load_param_fqns([nn.Linear(2, 2)]) == set()


def test_drop_skip_load_entries_removes_flagged_keys_and_rejects_unflagged_meta_tensors():
    outer = _model()
    state_dict = {
        **{f"experts.{k}": v for k, v in outer.experts.state_dict().items()},
        "head.weight": outer.head.weight,
    }
    assert set(_drop_skip_load_entries([outer], state_dict)) == {"head.weight"}
    with pytest.raises(RuntimeError, match="still on meta"):
        _drop_skip_load_entries([outer], {**state_dict, "stray": torch.empty(1, device="meta")})
    with pytest.raises(RuntimeError, match="flagged to skip"):
        _drop_skip_load_entries([outer], {"head.weight": outer.head.weight})


def test_require_no_skip_load_params():
    _require_no_skip_load_params(set(), "the full-state CPU load")
    with pytest.raises(NotImplementedError, match="full-state CPU load"):
        _require_no_skip_load_params({"experts.weight"}, "the full-state CPU load")
