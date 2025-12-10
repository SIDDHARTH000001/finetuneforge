---

# **Direct Preference Optimization (DPO) Training Example**

This repository demonstrates how to fine-tune a Large Language Model (LLM) using **Direct Preference Optimization (DPO)**.
DPO is a reinforcement-learning-free method to align models with human preferences by comparing **chosen** and **rejected** responses.

---

## **What is DPO?**

* Instead of maximizing reward via a separate RL loop, DPO **directly optimizes** the model to prefer responses rated better by humans.
* It uses a **reference model** for baseline comparison and encourages the trained model to:

  * **Increase** probability of preferred (chosen) responses
  * **Decrease** probability of less preferred (rejected) responses

---

## **Workflow**

1. **Prepare Preference Data** – Each example contains:

   * `prompt`
   * `chosen` response
   * `rejected` response
2. **Load Reference Model** – A frozen copy of the pre-trained base model.
3. **Compute Log-Probabilities** – For both `chosen` and `rejected` responses.
4. **Calculate Reward** – Difference in log-probabilities between current and reference models.
5. **Convert to Loss** – Using the DPO loss function:

   $$
   \mathcal{L} = -\log \sigma(\beta \cdot (r_{\text{chosen}} - r_{\text{rejected}}))
   $$
6. **Optimize** – Update model weights to improve preference alignment.

---

## **Features**

* **No RL loop** – Simpler and faster than PPO.
* **Supports LoRA** – Fine-tune only a small subset of parameters.
* **Custom Dataset Loader** – Easily adapt to your preference dataset.
* **Metrics Tracking** – Loss, perplexity, and preference accuracy.

---

---

## **Dataset Format**

Example JSONL:

```json
{"prompt": "Tell me a joke", "chosen": "Why did the chicken...", "rejected": "I don't know."}
{"prompt": "What is the capital of France?", "chosen": "Paris", "rejected": "London"}
```

---

## **References**

* [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
* [Hugging Face TRL Library](https://huggingface.co/docs/trl/index)

---

