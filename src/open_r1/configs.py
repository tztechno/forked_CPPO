# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# All params for CPPO,DRGRPO,RAFT,REINFORCE were set.

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The name to store runs under.")},
    )

    #######################################

    # setting for CPPO
    
    metric: Optional[str] = field(
        default='smallest',
        metadata={"help": ("What metrics are used for pruning? smallest, largest or random.")},
    )
    pruning: float = field(
        default=0.0,
        metadata={"help": ("Whether to use pruning")},
    )
    sample_num: int = field(
        default=0,
        metadata={"help": ("sample_num")},
    )
    allocation: bool = field(default=False, metadata={"help": "Generate first, then prune."})


    #######################################
    
    # from trl grpo_config.py 2025-04-27
    # DRGRPO specific parameter 2025-04-27

    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    
    #######################################
    
    # RAFT-specific parameters 2025-04-26
    policy_loss: str = field(
        default='none',
        metadata={
            "help": "Type of policy loss to use. Options: 'none', 'vanilla', 'plusplus'",
            "choices": ["none", "vanilla", "plusplus"]
        },
    )
    kl_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient for KL divergence loss."},
    )
    clip_epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon for clipping in plusplus policy loss."},
    )

    #######################################

    # REINFORCE-specific parameters 2025-04-28

    reinforce_variant: str = field(
        default="none",
        metadata={
            "help": "REINFORCE variant to use. Options: 'none', 'vanilla', 'plusplus'",
            "choices": ["none", "vanilla", "plusplus"]
        },
    )
    reinforce_pp_beta: float = field(
        default=0.1,
        metadata={"help": "Beta parameter for REINFORCE++ (weighting between original and normalized rewards)."},
    )
    use_reward_scaling: bool = field(
        default=True,
        metadata={"help": "Whether to scale rewards by a constant factor in REINFORCE algorithms."},
    )
    reward_scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Factor to scale rewards by in REINFORCE algorithms."},
    )
    normalize_advantages: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages in REINFORCE algorithms."},
    )


    #######################################


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
