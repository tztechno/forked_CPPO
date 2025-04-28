
---

2025-04-27

### This is forked repository from https://github.com/lzhxmu/CPPO

### Here, we will construct playground to try various newly announced RL methods in the open r1 environment. https://github.com/huggingface/open-r1

### GRPO and CPPO can be executed using the original CPPO environment.

### We have already added the argument for DRGRPO like HF TRL library, so DRGRPO and DRGRPO+CPPO can be executed in our modified CPPO environemnts.

### We have just prepared prototypes of RAFT vanilla and RAFT++, so we can use them.

### We are going to prepare also Reinforce vanilla, Reinforce++ and Reinforce_Rej.

---

### GRPO, CPPO, DRGRPO, DRGRPO+CPPO with gsm
use configs.py, grpo_gsm.py, grpo_trainer_gsm.py

### RAFTvanilla and RAFT++ with gsm
use configs.py, grpo_gsm_raft.py, grpo_trainer_gsm_raft.py

---

### The following literature is of interest:

#### CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models (cppo)
https://arxiv.org/abs/2503.22342

#### Understanding R1-Zero-Like Training: A Critical Perspective (drgrpo)
https://arxiv.org/abs/2503.20783

#### A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (raft)
https://arxiv.org/abs/2504.11343

#### REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models (reinforce)
https://arxiv.org/abs/2501.03262

---
