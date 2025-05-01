
---

2025-04-27

### This is forked repository from https://github.com/lzhxmu/CPPO

### Here, we will construct playground to try various newly announced RL methods in the open r1 environment. https://github.com/huggingface/open-r1

### GRPO and CPPO can be executed using the original CPPO environment.

### We have already added the argument for DRGRPO like HF TRL library, so DRGRPO and DRGRPO+CPPO can be executed in our modified CPPO environemnts. https://github.com/huggingface/trl/releases/tag/v0.16.0

### We have just prepared prototypes of RAFT vanilla and RAFT++, so we can use them (2025-04-27).

### We have just prepared prototypes of Reinforce vanilla, Reinforce++, so we can use them (2025-05-01).

---

### GRPO, CPPO, DRGRPO, DRGRPO+CPPO with gsm
use configs.py, grpo_gsm.py, grpo_trainer_gsm.py

### RAFTvanilla, RAFT++, REINFORCEvanilla, REINFORCE++ with gsm
use configs.py, grpo_gsm_raft.py, grpo_trainer_gsm_raft.py

---

### The following methods are of our interest:

#### CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models (cppo)
* https://arxiv.org/abs/2503.22342
* https://github.com/lzhxmu/CPPO
  
#### Understanding R1-Zero-Like Training: A Critical Perspective (drgrpo)
* https://arxiv.org/abs/2503.20783
* https://github.com/sail-sg/understand-r1-zero
  
#### A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (raft)
* https://arxiv.org/abs/2504.11343
* https://github.com/RLHFlow/Minimal-RL

#### A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (reinfirce)
* https://arxiv.org/abs/2504.11343
* https://github.com/OpenRLHF/OpenRLHF

#### DAPO: An Open-Source LLM Reinforcement Learning System at Scale (dapo)
* https://arxiv.org/abs/2503.14476
* https://github.com/ai-in-pm/DAPO

---
