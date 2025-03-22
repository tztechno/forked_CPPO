# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models


## Abstract
We introduces **Completion Pruning Policy Optimization (CPPO)** to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need for sampling multiple completions for each question. Our analysis reveals that the number of completions impacts model accuracy sublinearly yet increases training time multiplicatively, and not all completions contribute equally to policy training---their contribution depends on their relative advantage. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experiments on GSM8K datasets and Qwen2.5-1.5b-Instruct models demonstrate that CPPO accelerates reasoning model training by nearly **1.60$\times$** while maintaining the same performance as the original GRPO.

## Motivation

GRPO's policy objective function:
$$
    \mathcal{J}_{GRPO}(\theta) = 
    \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(o|q)}
    \Bigg\{  
    \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} 
    \Big\{ \min \Big[
    \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} A_{i},  \notag \\
    \text{clip} \big( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \big) A_{i} \Big] 
    - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] \Big\} \Bigg\}. \tag{1}
$$

The GRPO algorithm's training overhead scales linearly with the number of completions sampled per question. This is due to the need to compute predicted probabilities for the policy, reference, and old policy models across all completions. For instance, in DeepSeek-Math, using 64 completions requires 192 forward pass per question (64$\times$3), incurring significant computational costs. This raises two critical questions: 

**(1) How does the number of completions affect policy model accuracy? Does increasing completions always enhance performance?**

![image](./asset/analyze.png)

The number of completions impacts model accuracy **sublinearly** yet increases training time **multiplicatively**. Crucially, reducing completions to cut costs risks degrading reasoning capabilities, making it impractical.

**(2) Do all completions in a group contribute equally to training?**

**Not all completions contribute equally to policy training**---their contribution depends on their relative **advantage**.

The derivative of the GRPO's policy objective function in Eq.(1) with respect to the model parameters $\theta$ as:
$$
\begin{aligned}
 \nabla_{\theta} J_{GRPO}(\theta)=&\,\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \nabla_{\theta}\left(\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} A_i\right) \\
&\quad\quad\quad\quad - \beta\left(\nabla_{\theta} \frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}-\nabla_{\theta} \log \frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\right) \Bigg]\Bigg\} \\
%
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}\left(o_{i, t} | q, o_{i,<t}\right)}}A_i \\
&+ \beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}^{2}\left(o_{i, t} | q, o_{i,<t}\right)} - \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\right) \Bigg] \Bigg\} \\
%
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{  \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[ \frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} A_i  \\ 
&\quad\quad\quad\quad\quad\quad\quad + \beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)} - 1\right)\Bigg] \frac{\nabla_{\theta} \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}\Bigg\} \\
%
=&\mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Bigg[  \underbrace{\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} A_i}_{
\textit{Advantage-weighted probability ratio}} \\ 
&\quad\quad\quad\quad\quad + \underbrace{\beta\left(\frac{\pi_{r e f}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}-1\right)}_{\textit{KL divergence constraint}}   \Bigg] \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}_{
\textit{Policy model gradient}}\Bigg\}.
\end{aligned} \tag{2}
$$

 **(1) Advantage-weighted probability ratio term**  directly ties the contribution of each completion to its advantage. This term incentivizes the policy to prioritize actions with higher rewards, as the advantage function quantifies how much a given action improves expected returns relative to the baseline. By amplifying high-advantage completions and suppressing low-advantage ones, this term guides the policy optimization toward reward-aligned reasoning patterns.
%
**(2) KL divergence constraint term**  enforces stability by penalizing deviations from the reference model $\pi_{ref}$. However, this constraint is not inherently designed to shape the policy's reasoning patterns but rather ensures smooth updates during training.
%
**(3) Policy model gradient term** represents the gradient of the log-probability of the policy's predicted action with respect to the model parameters $\theta$.


Recent work by [Hu et al.](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero,) demonstrates that removing the KL divergence constraint does not impair the trained model's reasoning ability, as the policy's core reasoning patterns are primarily driven by the reward-aligned advantage term. Motivated by this insight, we approximate the policy objective's derivative as:

$$\begin{aligned}
\nabla_{\theta} J_{GRPO}(\theta) &  \approx  \, \mathbb{E}_{\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]} \\
&\Bigg\{ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \sum_{t=1}^{\left|o_{i}\right|} \Big[ \underbrace{\frac{\pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}{\pi_{\theta_{old}}\left(o_{i, t} | q, o_{i,<t}\right)} }_{\substack{\textit{Probability ratio} \\ \textit{(Post-forward)}}}\cdot \underbrace{A_i}_{\substack{\textit{Advantage}\\\textit{(Prior-forward)}}}  \Big] \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(o_{i, t} | q, o_{i,<t}\right)}_{\substack{\textit{Policy model gradient}\\ \textit{(Post-forward)}}} \Bigg\}, \tag{3}
\end{aligned}$$

effectively decoupling the optimization from KL regularization while retaining the reward-driven learning signal.


To better understand this formulation, we decompose the advantage-weighted probability ratio term into the **Probability ratio** term and the **Advantage** term. For a completion that significantly contribute to the policy update, all the new three components in Eq.(3) must be non-negligible. A  near-zero or zero value in any of these components would render the overall contribution minimal or nonexistent.

From a computational timing perspective, these components can be categorized as:
(1) the probability ratio and policy model gradient are post-forward information, meaning they can only be computed after the policy's forward pass.
(2) The advantage term, however, represents prior-forward information that can be calculated before the policy's forward computation.


Given our objective to accelerate GRPO training, we focus on leveraging this prior-forward information. By evaluating the advantage term before the forward pass, we can make an informed decision about whether to process a completion through the policy model. 


## Completions Pruning Policy Optimization

![image](./asset/CPPO.png)

The pipeline of the CPPO algorithm is as follows:

(1) The old-policy model samples a group of completions for each question.
(2) The reward function computes the reward for each completion via :
$$ r_i = R_{format}(o_i) + R_{accuracy}(o_i), $$

where 

$$
\begin{aligned}
    R_{\text{format}}(o_i) &= 
    \begin{cases}
        1, & \text{if } o_i \text{ follows the correct format}, \\
        0, & \text{otherwise}.
    \end{cases} \\
    R_{\text{accuracy}}(o_i) &= 
    \begin{cases}
        2, & \text{if } o_i \text{ directly matchs the correct answer}, \\
        1.5 & \text{if } o_i \text{ matchs the correct answer after regular parsing}, \\
        0, & \text{otherwise}.
    \end{cases}
\end{aligned}
$$

(3) The relative advantage of each completion is calculated according to:

$$
A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \dots, r_G\})}
    {\mathrm{std}(\{r_1, r_2, \dots, r_G\})}. 
$$

(4) CPPO retains $k = \lfloor G \times (1 - P)\rfloor $ completions with highest absolute advantages, $P$ is pruning rate and $G$ is the compeletion number.

(5) The policy model is updated based on the selected completions.


## Parallel Processing through Dynamic Completion Allocation
![alt text](./asset/allocation.png)

A single device can process a maximum of $B$ question per batch, with each question generating $G$ candidate completions. After pruning, the total number of retained completions per device reduces to $B \times k$, resulting in suboptimal GPU utilization and underleveraged parallel computing capabilities.

To address this inefficiency, we dynamically allocate pruned completions from additional questions into the device's processing pipeline, as illustrated in the figure above. This strategy ensures that each device operates at full capacity by continuously populating its memory with high-quality completions derived from both the original and newly introduced questions. Critically, all newly incorporated completions undergo the same rigorous pruning process to maintain consistency and relevance.
## Key Results

| Model      | Method | Group Size |Pruning Rate | Accuracy | Time | Accelerate Ratio
| ----------- | ----------- |- |-|-|-|-|
| Qwen2.5-1.5B-Instruct      | -       |-|-|-|-|-
| Qwen2.5-1.5B-Instruct      | GRPO    |16 |0%| |   |
| Qwen2.5-1.5B-Instruct      | GRPO    | 8 |0% |  |   |
| Qwen2.5-1.5B-Instruct      | CPPO    |16 |50%|    |   |
## To Reproduce

### 1. Prepare the environment according to [Open R1](https://github.com/huggingface/open-r1).
### 2. Training:
#### GRPO
```bash
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1  src/open_r1/grpo_gsm.py \
    --config recipes/gsm8k/config_simple_rl.yaml \
    --output_dir=/data/CPPO \
    --save_strategy='best' \
    --num_generations=8 --metric='smallest' --pruning=0.5  --filling 
```
#### CPPO
```bash
accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1  src/open_r1/grpo_gsm.py \
    --config recipes/gsm8k/config_simple_rl.yaml \
    --output_dir=/data/GRPO \
    --save_strategy='best' \
    --num_generations=16 --num_iterations=2 2>&1 
```
### 3. Evaluation:
```bash
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1  src/open_r1/eval_gsm.py \
    --config recipes/gsm8k/eval.yaml \
    --output_dir=/data/eval \
    --per_device_eval_batch_size=16 \
    --max_completion_length=1024 \
    --model_name_or_path=MODEL_CKPT_PATH \
    --num_generations=16 --num_train_epochs=0
```


## Acknowledgments
We are very grateful to the [Open R1](https://github.com/huggingface/open-r1) teams for createing awesome repo.
