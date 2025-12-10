# 🔧 FinetuneForge

A curated collection of **fine-tuning strategies for LLMs and GenAI models**, built for experimentation, benchmarking, and extensibility.  
This repo brings together multiple approaches — from classical distillation to cutting-edge RLHF — with modular code and custom loss functions.

---

## 📌 Included Approaches

- **Knowledge Distillation**: Compress large models into smaller, efficient ones.
- **Prompt Tuning**: Lightweight adaptation via soft prompts.
- **QLoRA + Custom Loss**: Efficient low-rank adaptation with structured JSON response objectives.
- **RLHF Variants**:
  - **DPO (Direct Preference Optimization)**
  - **PPO (Proximal Policy Optimization)**
  - **GRPO (Group Relative Preference Optimization)**
- **Mixture of Experts (MoE) Fine-tuning**: Scaling with expert routing.

---

## 🎯 Goals
- Provide **minimal, reproducible examples** for each fine-tuning method.
- Benchmark trade-offs: latency, throughput, quality.
- Enable **custom loss functions** for structured outputs (e.g., JSON).
- Serve as a **playground for future approaches** (new RLHF variants, hybrid tuning, etc.).

---

## 🛠️ Structure
