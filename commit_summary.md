

---

# REINFORCE

    [configs.py]

    #######################################
    
    [grpo_trainer_gsm_raft.py]


---

# RAFT

    [configs.py]
    
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
    
    [grpo_trainer_gsm_raft.py]    
    
    # RAFT L.186-188,L458-469,L.922-950 2025-04-26

        # RAFT-specific initialization
        if hasattr(self.config, 'policy_loss') and self.config.policy_loss in ['vanilla', 'plusplus']:
            self._init_raft_components()
    
        def _init_raft_components(self):
        """RAFT用のコンポーネント初期化"""
        if self.config.policy_loss in ['vanilla', 'plusplus']:
            # 報酬正規化用バッファ
            self.reward_buffer = {
                'min': torch.tensor(float('inf')),
                'max': torch.tensor(float('-inf'))
            }
            # クリッピング範囲のデフォルト値設定
            if not hasattr(self.config, 'clip_epsilon'):
                self.config.clip_epsilon = 0.2  # デフォルト値

        # 2. RAFT用の損失計算
        if self.config.policy_loss in ['vanilla', 'plusplus']:
            rewards = inputs.get('raw_rewards', advantages)  # 生の報酬値を取得
            weights = self._normalize_rewards(rewards).unsqueeze(1)  # 報酬を[0,1]範囲に正規化
            
            if self.config.policy_loss == 'plusplus':
                # クリッピングを適用 (1-ε, 1+εの範囲に制限)
                weights = torch.clamp(weights, 
                                   1-self.config.clip_epsilon, 
                                   1+self.config.clip_epsilon)
            
            # 報酬重み付けされた損失
            per_token_loss = -per_token_logps * weights
            
            # メトリクス記録
            self._metrics[mode]["reward_weight_mean"].append(weights.mean().item())
            self._metrics[mode]["reward_weight_std"].append(weights.std().item())

---

# DRGRPD

    [configs.py]
    
    # from trl grpo_config.py 2025-04-27
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
    
    [grpo_trainer_gsm.py]   
    
    # scale_rewards setting added for drgrpo 2027-04-27
    # L.449, L.899-901 
    
        #### for drgrpo 2025-04-27
        self.scale_rewards = args.scale_rewards

        #### for drgrpo 2025-04-27
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)


---

# CPPO

    [configs.py]

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
    
    [grpo_trainer_gsm.py]   




---
