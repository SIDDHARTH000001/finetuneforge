# PPO (Proximal Policy Optimization)

* ***These tasks get really difficult to design reward systems for which is where RLHF comes in. What if, instead  of designing a system, we simply have a human make suggestions? A human knows what a backflip is after all. The human will act like a tutor picking attempts it likes more as the bot is training. That’s what RLHF is and it works really well. Applied to LLMs, a human simply looks at generated responses to a prompt and picks which one they like more***

* ***Reinforcement Learning with Human Feedback substitutes a loss function for a reward model and PPO, allowing the model a much higher ceiling for learning trends within the data, including what is preferred as an output, as opposed to what completes the task***

* ***The key idea behind PPO is that it optimizes the policy while keeping updates within a “proximal” (or close) range of the previous policy to avoid large, destabilizing changes.***

* ***The process goes like you give a llm a prompt it generates two different response `A & B` for the given prompt, you give these two response to User and user select which one is better ,suppose 'A' is the one which user selects, then we take these two reponse and try to train one Reward Model which learn to give the score A>B so far we have done the first part,Next is just take the another prompt and give it to model and now the model give a reposen called `C` in this time our model is in .eval() model or there we take two model one for reference which is base model another for training,meaning there will be no backpropagation so its just a raw detached log probabilites from the llm, now we give this `C` to reward model and model gives the reward here, so here one thing need to be considered that because model can give any raw value i.g - 10323, 132, 2324..etc so to include this in the back propagation if we include this raw reward this will leads to large fluctuation in model weights so we just try to use some base value like score like 132 - 100 which give 32 as result so this just kind of now normalized with direction from base value in +ve or -ve direction,` Advantage measures how much better the model performed compared to what it usually does.`, `this is called advantage` and this is used in calculation further, now the main part is how to involve this `advantage_score` to our loss because it can not be returned directly becasue it comes from diff model and backpropagation does not get invovled in this because it is the result of some other reward models output and also to include this directly is very risky because we know llm are probabilistic models and do not provide same output each time so what if A or B are not stable output like A is very less likely but it came this time or what if B has the same issue, for that we use this PPO technique,which stands for proximal policy optimization meaning we updated the model while being in the proximity of base llm not making drastic changes, now just we give the same prompt again to the model but now in .train() mode so we get gradient in backpropagation and main reason to make model update stable becasue the prompt is same so model should give almost similar logits/log probabilities (as log_prob_new) otherwise in case of if the 1st and 2nd time it will be very different than we will endup changing model drastically, once the we have both log probabilties log_prob_old and log_prob_new we just calculate the ratio of these probabilties so prob_A/prob_B but we get these in form of log_prob not prob so so far we have log_ratio which need to be converted into ratio so we do exponential of both side than ratio becomes ratio = e^(log_prob_A/ log_prob_B), now we take this ratio and multiply with advantage which become `ratio * advantage` so far these values might contain drasticlly difference between A and B, meaning A might be not normal behaviour of A or but B is then we will push our model toward A which is very outliear and same goes for B, so to make this stable we clipped the larger differnce/ratio for that we perform `clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage` this give the clipped version of it making sure no drastic changes, next we just take the min form both of the clipped and unclipped (to keep the orignal values from unclipped) like this ppo_loss = -torch.mean(torch.min(unclipped, clipped)) and this becomes our PPO loss based on this we update our model weights. now the things is over taining with this PPO policy things will drift very little each time but over the time things might get too much drifted for that we use KL divergence where we take current log_prob_new and take the another frozen copy of same base SFT-model, we take the reposen from it for same prompt and perform KL-divergence this make sure we dont go too far from the Base-SFT or any catestrophic forgetting does not happen, There are other terms also in loss part like vf_coef which is just `loss * vf_coef` which tell how aggresive we want to consider the loss, another is entropy(optional but used widely) =>***

* ***so in enptory part we know for suppose batch of 8 we will have 8 llm output sentense suupose all are of length 10 so now we have 8 X 10 now from llm we know each token in this 8 sentence have the distribution over vocab so it will be 8 x 10 x vocab_size, now in entorpy we use this token distribution over vocab for each token we will do `prob x ln(prob)` this is because if prob for 1 token is very high suppose .99 then it means model was so much confidant on this but suppose two tokens have .24, .25 this prob and we do this entropy this will give high -ve value for both for current token so when we sum it up it shows a larger value showing model had multiple choice or it has exploration option for that token , after doing sum on this last dim whihc remove vocab_size it will just remain batch X seq_len where each seq is n number showing the entropy total, so we perform at the end overall mean which is over all the token includeing current seq and across batch all the seq to show per token ow much avg entory was there if it is high we include this also to drive model to explore those other possibitlities ***
***

### ✅ **1. Where does human feedback come in?**

*   **Before RLHF**, we do **Supervised Fine-Tuning (SFT)** using curated human-written examples.
*   Then, **human feedback is collected** by showing humans **pairs of model responses** to the same prompt and asking:
    > “Which one do you prefer?”
*   These **preferences** are **NOT used to train the language model directly**.  
    Instead, they are used to **train the Reward Model**.

***

### ✅ **2. How is the Reward Model trained?**

*   The reward model takes:
    *   **Prompt**
    *   **Candidate response**
*   And predicts a **score** that correlates with human preference.
*   Training:
    *   If humans prefer Response A over Response B, the reward model learns to give A a higher score than B.
*   So yes, when ChatGPT asks you to pick one of two answers, that data goes into **reward model training**, not PPO yet.

***

### ✅ **3. Where does PPO fit in?**

*   After the reward model is trained, we use **PPO** to fine-tune the language model:
    *   Generate responses from the current policy (the model).
    *   Score them using the **reward model**.
    *   Update the model using PPO to **maximize reward** while staying close to the original SFT model (via KL penalty).
*   PPO is the **optimization algorithm** that uses the reward model’s scores as the signal.

***

### ✅ **Pipeline Summary**

1.  **SFT**: Train on human-written examples.
2.  **Collect human preferences**: Show two answers → pick one.
3.  **Train Reward Model**: Predict preference scores.
4.  **RL with PPO**:
    *   Model generates responses.
    *   Reward model scores them.
    *   PPO updates the model to maximize reward.

***

---

# Correct RLHF PPO pipeline

## Step 1: Collect human preference data

For each prompt:

* LLM generates two outputs A and B.
* Human chooses which one is better.

This pairs the outputs into preference data.

---

## Step 2: Train a reward model using pairwise ranking loss

Reward model R learns:

```
R(A) > R(B) if user prefers A
```

This is done with a Bradley Terry loss:

```
loss = -log(sigmoid(R(A) - R(B)))
```

This reward model is now used to score new LLM outputs.

You understood this part correctly.

---

## Step 3: Rollout phase

We freeze reward model and SFT policy.

1. For a new prompt:

   ```
   C = policy.generate(prompt)    (policy in eval mode)
   ```
2. Compute reward:

   ```
   reward = R(prompt, C)
   ```

The policy is not trained here. Only reward is collected.

You understood this part correctly.

---

# Step 4: Why we need an advantage, what it actually is

This part is the most misunderstood area.

### Correct definition:

Advantage is:

```
A = reward - baseline
```

Where baseline is usually a value function V estimated by a critic model.

This is not the old reward model.
It is part of PPO: policy gradient needs an estimate of:

```
how much better was that action than usual
```

### Why we need advantage:

* Raw reward magnitude causes unstable gradients.
* Advantage captures improvement relative to expectation.
* Adv reduces variance and makes PPO stable.

Your explanation about needing a normalized signal is correct, but the reason is the baseline and value function, not arbitrary subtractions.

---

# Step 5: Why we use old and new log probabilities

This part needs careful correction.

### Old probability

Stored during rollout:

```
log_prob_old
```

### New probability

Computed during training:

```
log_prob_new
```

### Key point:

We compare how much the policy changed:

```
ratio = exp(log_prob_new - log_prob_old)
```

Very important:
It is subtraction, not division of logs.

So:

```
ratio = exp(log_prob_new - log_prob_old)
```

This gives:

```
ratio = prob_new / prob_old
```

Your explanation had this mixed up.

---

# Step 6: PPO objective

Unclipped objective:

```
loss_unclipped = - ratio * advantage
```

Clipped objective:

```
loss_clipped = - clamp(ratio, 1 - eps, 1 + eps) * advantage
```

Final objective:

```
loss = mean(max(loss_unclipped, loss_clipped))
```

This ensures:

* Model can improve if ratio is near 1.
* Model cannot make huge jumps.

This part you understood correctly.

---

# Step 7: KL penalty to prevent drifting

Your explanation here is correct in the core idea, but let me phrase it precisely.

We compute KL divergence between:

* Current policy (policy new)
* Frozen SFT policy (policy ref)

```
KL = KL(policy_new || policy_ref)
```

KL is added to the loss:

```
loss_total = ppo_loss + beta * KL
```

This ensures policy stays close to SFT and prevents destructive changes.

---

# Step 8: Iterate this process

Over many iterations:

* Policy becomes better aligned with human preferences
* Does not drift far due to KL penalty
* PPO clipping keeps updates stable

Your final mental picture is consistent with reality.

---

# Final corrected summary of your understanding

Here is your idea rewritten correctly and compactly.

1. Generate A and B for prompt.
2. Human chooses A.
3. Train reward model on pairwise preference (R(A) > R(B)).
4. Freeze reward model.
5. Sample new response C from SFT model.
6. Score C with reward model.
7. Compute advantage:

   ```
   A = reward - baseline (value function)
   ```
8. Compute likelihood ratio:

   ```
   ratio = exp(log_prob_new - log_prob_old)
   ```
9. Compute PPO clipped objective:

   ```
   loss = mean(min(ratio*A, clipped_ratio*A))
   ```
10. Add KL penalty to keep model near SFT:

    ```
    loss += beta * KL(policy_new || policy_ref)
    ```
11. Backprop this loss.
12. Repeat.

This is PPO based RLHF.

---

# If you want next

I can also give you:

* A step by step diagram of PPO RLHF
* Full formulas in clean math
* Annotated PyTorch style pseudocode of PPO for LLMs
* Explanation of why KL penalty is essential for LLM stability
* How Rejection Sampling Fine Tuning differs from PPO based RLHF

Tell me what you want.

* ***While very powerful, it likely won’t stick around for very long. The reason is that it is incredibly computationally expensive for a result that is only incrementally better, especially a result that can be achieved and matched by higher-quality datasets with supervised learning approaches***

* ***There are some other issues with RLHF like it requires hiring domain experts to evaluate and provide the human feedback. Not only can this get expensive, but this can also lead to privacy concerns since these reviewers would need to look at actual traffic and user interactions to grade them. To combat both of these concerns, you could try to outsource this directly to the users asking for their feedback, but it may end up poisoning your data if your users have ill intent or are simply not experts in the subject matter in which case they might upvote responses they like, but aren’t actually correct. This gets to the next issue, even experts have biases. RLHF doesn’t train a model to be more accurate or factually correct, it just simply trains it to generate human acceptable answers***

* ***RLHF depends on human feedback data collected from prompts and responses.If the feedback data over time comes from similar prompts or narrow domains (e.g., evaluation sets or repeated user queries), then:The reward model learns a very specific preference pattern PPO fine-tuning repeatedly reinforces those patterns***


* **how PPO calculates these terms during training in Python**

***

### ✅ PPO Loss Calculation in Python (Conceptual Flow)

When you call:

```python
stats = ppo_trainer.step(queries, responses, rewards)
```

TRL internally computes:

***

#### **1. Compute log probabilities**

```python
logprobs = policy_model.get_logprobs(queries, responses)
ref_logprobs = ref_model.get_logprobs(queries, responses)
```

*   These are the log probabilities of the generated tokens under the **current policy** and **reference model**.

***

#### **2. Compute KL divergence**

```python
kl_div = (logprobs - ref_logprobs)
```

*   Used for KL penalty and adaptive learning rate.

***

#### **3. Compute ratio for PPO clipping**

```python
ratio = torch.exp(logprobs - old_logprobs)
```

*   `old_logprobs` are stored from the previous policy before update.

***

#### **4. Compute Advantage**

```python
advantages = rewards - values.detach()
```

*   `values` are predicted by the **value head**.
*   Rewards come from your `reward_fn`.

***

#### **5. Policy Loss**

```python
cliprange = 0.2
policy_loss = -torch.mean(torch.min(
    ratio * advantages,
    torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
))
```

***

#### **6. Value Loss**

```python
value_loss = torch.mean((values - rewards) ** 2)
```

*   Weighted by `vf_coef`:

```python
value_loss = vf_coef * value_loss
```

***

#### **7. Entropy Loss**

```python
entropy = -torch.mean(torch.sum(probs * logprobs, dim=-1))
entropy_loss = -entropy_coef * entropy
```

***

#### **8. Total Loss**

```python
loss = policy_loss + value_loss + entropy_loss
loss.backward()
optimizer.step()
```

***

### ✅ In TRL (`ppo_trainer.step()`):

*   All these steps are done internally.
*   You only provide:
    *   `queries` (prompt tokens)
    *   `responses` (generated tokens)
    *   `rewards` (from your reward function)

TRL handles:

*   Computing logprobs, KL, clipping, advantage, and updating the model.

***

## **entropy in PPO** 

***

### ✅ What is Entropy in PPO?

In reinforcement learning, **entropy** measures the **uncertainty** or **randomness** in the policy’s action distribution.

*   A **high entropy** means the policy is exploring (actions are more random).
*   A **low entropy** means the policy is exploiting (actions are more deterministic).

Adding an **entropy bonus** encourages exploration, preventing the policy from collapsing too quickly into a narrow set of actions.

***

### ✅ What are `probs` and `logprobs`?

*   **`probs`**: The probability distribution over tokens (actions) predicted by the policy model.  
    For each token in the sequence, the model outputs a softmax over the vocabulary.

*   **`logprobs`**: The logarithm of those probabilities (log-softmax).  
    Log probabilities are used because they are numerically stable and convenient for KL divergence and likelihood calculations.

So for a batch of sequences:

```python
probs.shape      # [batch_size, seq_len, vocab_size]
logprobs.shape   # [batch_size, seq_len, vocab_size]
```

***

### ✅ Why `entropy = -torch.mean(torch.sum(probs * logprobs, dim=-1))`?

This is the formula for **Shannon entropy**:

$$
H(p) = - \sum\_{i} p\_i \log p\_i
$$

*   For each token position, we compute:
    $$ \text{entropy per token} = - \sum\_{vocab} p(v) \log p(v) $$
*   `torch.sum(..., dim=-1)` sums over the vocabulary dimension.
*   Then `torch.mean(...)` averages across tokens and batch.

So:

*   **Inner sum** → entropy for each token.
*   **Outer mean** → average entropy across all tokens in the batch.

***

### ✅ Why `entropy_loss = -entropy_coef * entropy`?

*   We **maximize entropy** (encourage exploration), but since we minimize loss, we add it as a **negative term**:
    $$ \text{loss} = \text{policy\_loss} + \text{value\_loss} - \beta H(p) $$
*   `entropy_coef` (often called `beta`) controls **how strongly we encourage exploration**.

***

### ✅ Intuition:

*   If `entropy_coef` is **high**, the model will stay more random for longer (exploration).
*   If `entropy_coef` is **low**, the model will become deterministic faster (exploitation).

***

🔥 **Summary in one line:**  
`entropy` here is the average uncertainty of the policy’s token predictions. We subtract it (scaled by `entropy_coef`) from the total loss to keep the policy from collapsing too early.

***
