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

"""Utilities for stable logical parameter names."""

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX


def canonical_parameter_fqn(name: str) -> str:
    """Remove activation-checkpoint wrapper components from a parameter FQN.

    PyTorch exposes ``CheckpointWrapper`` as a registered child module through
    ``named_parameters()``, but strips the same component from state-dict keys.
    Canonicalizing name-sensitive consumers keeps their logical names aligned
    with the stable state-dict contract.

    Args:
        name: Module-qualified parameter name.

    Returns:
        The logical parameter name without checkpoint-wrapper components.
    """
    return name.replace(_CHECKPOINT_PREFIX, "")


# Module attribute: a set of that module's parameter names that the module loads itself (e.g. NF4-resident
# experts streamed from a floating-point HF checkpoint). ``to_empty_parameters_only`` keeps those parameters
# on meta and ``Checkpointer.load_model`` leaves them out of the base load. Kept on the module, not on the
# parameter: ``nn.Module._apply`` swaps DTensor parameters with ``torch.utils.swap_tensors``, which also
# swaps the Python ``__dict__`` and would drop a per-parameter flag.
SKIP_CHECKPOINT_LOAD_ATTR = "_skip_checkpoint_load_params"
