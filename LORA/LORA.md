## **PEFT & LoRA**:

***
* ***this lora is highly inspired from SVD where we decompose any matrix in three smaller matrix `U, V and E` where U is hold the eigan vectors in data domian capturing the Dataset pattern and V hold the vectos in feature domain capturing the realtion of features and E holds the eigan values, so this we take in considerration and because deep leaning model can learn/approximate anything so we try to learn these two matrix u and V here of rank r which reduces the number of paramters and make it efficient ,suppose we have 100 layer deeplearning model and each having 100 weights(not 100 neurons), so we can put this in a matrix of 100 x 100 and try to decompse into 100 x r and r x100 where r is the rank, r holds the eigan values in diagnal of sigma matrix, SO the SVD part is just for inspiration that we can reduce the any matrix into smaller submatrix so here we do not calculate the whole SVD but just two decompsed matrixs***


* ***If you’re training a 1 billion parameter model, you should train it on 20 billion tokens. This number comes from the Chinchilla paper[6] . If you have fewer tokens than that, you will run the risk of overfitting.***

---

### 📌 Rule of Thumb for Fine‑tuning
- **Token requirement**: You should have at least ~0.000001 × (model parameters) in tokens.  
  - Example: For a **1 billion parameter model**, minimum ≈ **10k tokens**.  

- **Origin of rule**: This is based on practical experience, not a formal paper, but it’s intuitive.  

---

### ⚖️ Impact of Too Few Tokens
- If you have **fewer than 1/100,000 of your model parameters in tokens**, fine‑tuning will likely have little effect.  
- In such cases, the model won’t adapt meaningfully to your dataset.  

---

### 🚀 Alternatives to Full Fine‑tuning
- **LoRA (Low‑Rank Adaptation)**: Efficient adapter‑based fine‑tuning that requires fewer tokens and less compute.  
- **Retrieval‑Augmented Generation (RAG)**: Augment the model with external knowledge sources instead of retraining.  

---

### ✅ **PEFT & LoRA Key Points**

*   **LoRA (Low-Rank Adaptation)**:
    *   A breakthrough technique for efficient fine-tuning.
    *   Allows adapting a model to new tasks **without changing original weights**.
    *   Saves memory and compute cost significantly.

*   **Why LoRA is important**:
    *   Enables fine-tuning for multiple domains without interference.
    *   Common in diffusion models (e.g., text-to-image LoRAs).

*   **Core idea**:
    *   Instead of retraining full weight matrices, train and store only the **difference** between base model and target domain.
    *   Achieved using **Singular Value Decomposition (SVD)**.

*   **Mathematical intuition**:
    *   Normal weight update:  
        $$ W = W + \Delta W $$
    *   LoRA decomposes:  
        $$ \Delta W = W\_a \times W\_b $$
        *   If original weight matrix is 100×100:
            *   ( W\_a = 100 \times c )
            *   ( W\_b = c \times 100 )
            *   Where ( c < 100 ) (low rank).

*   **Rank (R in LoRA)**:
    *   Determines complexity and memory usage.
    *   Higher rank → closer to original accuracy but less memory saved.
    *   Lower rank → more memory savings but less accuracy.
    *   Rank can be estimated algorithmically (Eigenvectors) or approximated based on task complexity.

*   **Practical insight**:
    *   If task is simple → use lower rank.
    *   If task is complex → use higher rank for better performance.

***

🔥 **Summary in one line**:  
LoRA fine-tunes models by learning low-rank matrices that approximate weight updates, making multi-domain adaptation lightweight and efficient.

***
* `TrainingArguments` and explain when you’ll see the **loss printed** during training:

---

### ⚙️ Parameters Explained

- **`output_dir="./qwen-qlora-it"`**  
  Where checkpoints, logs, and final model weights will be saved.

- **`per_device_train_batch_size=2`**  
  Number of samples per GPU (or CPU) per step. With small batch sizes, you often use gradient accumulation to simulate larger batches.

- **`gradient_accumulation_steps=8`**  
  Instead of updating weights every step, accumulate gradients for 8 steps before applying an update.  
  → Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps`.  
  In your case: `2 × 8 = 16`.

- **`num_train_epochs=3`**  
  How many times the model will iterate over the entire dataset.

- **`learning_rate=2e-4`**  
  The optimizer’s step size. Controls how fast weights are updated.

- **`fp16=True`**  
  Enables mixed precision (16‑bit floating point). Speeds up training and reduces memory usage.

- **`logging_steps=20`**  
  Every 20 steps, the trainer will log metrics (like loss).  
  ⚡ This is when you’ll see the **loss printed** in your console.

- **`save_steps=200`**  
  Every 200 steps, the trainer saves a checkpoint to `output_dir`.

- **`report_to="none"`**  
  Disables reporting to external loggers (like TensorBoard, WandB). Logs only to stdout.

---

### 🔎 When Loss is Printed
- Loss is printed **every `logging_steps`** (here, every 20 steps).  
- Example console output:
  ```
  Step 20: loss = 2.31
  Step 40: loss = 2.05
  ...
  ```
- If you want more frequent logging, lower `logging_steps` (e.g., `logging_steps=5`).  
- If you want to see loss per epoch, you can also add `evaluation_strategy="epoch"` in `TrainingArguments`.

---

### ✅ Summary
- Your setup trains with an effective batch size of 16, for 3 epochs.  
- Loss will be printed every 20 steps.  
- Checkpoints will be saved every 200 steps.  
- All logs go to stdout since `report_to="none"`.

---
## As an Output

* ***the base model stays frozen**, and only the **adapter layers** are trained and saved. So each `checkpoint-*` folder contains:

- The **adapter weights** (not the full model)
- The **optimizer state**, RNG state, scheduler, etc.
- Metadata like `training_args.bin` and `trainer_state.json`

---

### 🔍 What’s adapter-specific in each checkpoint

Here’s what matters for **swapping adapters**:

| File | Purpose | Adapter-specific? |
|------|---------|-------------------|
| `adapter_model.safetensors` | Trained adapter weights | ✅ Yes |
| `adapter_config.json` | PEFT config (e.g., LoRA rank, target modules) | ✅ Yes |
| `optimizer.pt`, `scaler.pt`, `scheduler.pt` | Training state (not needed for inference) | ❌ No |
| `trainer_state.json`, `training_args.bin` | Logging and metadata | ❌ No |

So if you want to **swap adapters**, you only need:
- `adapter_model.safetensors`
- `adapter_config.json`

---

### ✅ How to load a specific adapter checkpoint

Assuming you’ve already loaded the base model:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", load_in_4bit=True)

# Load adapter from checkpoint-600
adapter_path = "./checkpoint-600"
model = PeftModel.from_pretrained(base_model, adapter_path)
```

This will:
- Load the adapter weights from `checkpoint-600`
- Merge them with the frozen base model
- Let you run inference or continue training with that adapter

---

### 🧠 Why this is powerful
- You can train multiple adapters on different tasks or domains.
- Swap them in and out without touching the base model.
- Keep your base model on disk once, and just load adapters as needed.

---

* ***imbalanced multi‑GPU utilization**. Let’s break down why one GPU is pegged at ~80% while the other hovers between 16–40%.

---

### 🔎 Why one GPU is underutilized
1. **Model fits on one GPU**  
   - With `device_map="auto"`, Hugging Face Accelerate will place the entire model on a single GPU if it fits.  
   - The second GPU may only be used for small tasks (like optimizer state or offloading), which explains the low utilization.

2. **Data Parallelism vs Model Parallelism**  
   - **Data Parallelism (DDP)**: Each GPU gets a full copy of the model, and batches are split. This keeps both GPUs busy.  
   - **Model Parallelism (device_map)**: Layers are split across GPUs. If the split isn’t balanced, one GPU does more work.  
   - By default, your setup is leaning toward model parallelism with uneven load.

3. **Gradient accumulation**  
   - With small batch sizes (`per_device_train_batch_size=2`), gradient accumulation reduces per‑step GPU load. One GPU may idle while waiting for the other to finish.

4. **I/O bottlenecks**  
   - If the data loader or CPU preprocessing is slow, one GPU can stall while the other waits for batches.

---

### ✅ How to fix / balance GPU usage
- **Force balanced sharding**  
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      load_in_4bit=True,
      device_map="balanced",
      torch_dtype=torch.float16
  )
  ```
  → This distributes layers more evenly across both GPUs.

- **Use `accelerate` or `torchrun` for DDP**  
  Launch training with:
  ```bash
  accelerate launch train.py
  ```
  or
  ```bash
  torchrun --nproc_per_node=2 train.py
  ```
  → Each GPU gets a full copy of the model, and batches are split automatically.

- **Increase batch size if possible**  
  Larger batches keep both GPUs busier. If memory is tight, rely on gradient accumulation.

- **Check data pipeline**  
  Ensure your dataset preprocessing isn’t starving the GPUs. Use `num_workers` in `DataLoader` to parallelize CPU work.

---

### 🧠 Summary
- One GPU is underutilized because the model fits on a single GPU and isn’t being evenly sharded.  
- To fix: use `device_map="balanced"` for model parallelism, or run with `accelerate/torchrun` for true data parallelism.  
- Also check batch size and data pipeline efficiency.

---


* **`device_map="auto"`** (model parallelism) and **data parallelism (DDP)**.  

---

### 🔎 Why `device_map="auto"` ≠ Data Parallelism
- `device_map="auto"` tells Hugging Face/Accelerate to **shard the model layers across GPUs** if needed.  
- That’s **model parallelism** — splitting one model across devices.  
- **Data parallelism** means **each GPU gets a full copy of the model**, and the training batch is split across GPUs.  
- You can’t get data parallelism just by setting `device_map="auto"` — you need to launch training with a distributed runner (`torchrun` or `accelerate`).

---

### ✅ How to enable Data Parallelism
1. **Keep model load simple** (no device_map):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       load_in_4bit=True,
       torch_dtype="auto"
   )
   ```
   → This way, the model is not sharded — it’s replicated.

2. **Launch training with multiple processes**:
   - With `accelerate`:
     ```bash
     accelerate launch train.py
     ```
   - With PyTorch DDP:
     ```bash
     torchrun --nproc_per_node=2 train.py
     ```
   → Each process runs on one GPU, with a full copy of the model. The batch is split automatically.

3. **Trainer handles distribution**  
   Hugging Face `Trainer` integrates with Accelerate. If you launch with `accelerate` or `torchrun`, it will automatically use **DistributedDataParallel (DDP)** under the hood.

---

### ⚡ Key difference in configs
- **Model parallelism** → `device_map="auto"` or `"balanced"`.  
- **Data parallelism** → no `device_map`, but launch with `accelerate`/`torchrun`.  

---

### ✅ Summary
- `device_map="auto"` = model parallelism (layers split).  
- Data parallelism = replicate model on each GPU, split batches.  
- To switch: remove `device_map`, and launch training with `accelerate` or `torchrun`.  

---

---

## 🔎 The logic behind `-100` for the prompt part

- **CrossEntropyLoss in Hugging Face**:  
  By default, the loss is computed over every token in `labels`.  
  Hugging Face’s `Trainer` uses `ignore_index=-100`. That means any position in `labels` set to `-100` is **ignored** when computing the loss.

- **Prompt vs Response distinction**:  
  - The **prompt** is the conditioning context (what you feed the model).  
  - The **response** is what you want the model to learn to generate.  
  If you don’t mask the prompt, the model will waste capacity trying to “predict” the prompt tokens — which it already sees as input. That’s redundant and hurts training.

- **Masking with `-100`**:  
  By setting the prompt tokens in `labels` to `-100`, you tell the loss function:  
  *“Don’t backpropagate on these positions. Only learn from the response tokens.”*

---

## ✅ Example

Suppose your training pair is:

```
Prompt: "Q: What is the capital of Japan?\nA:"
Response: " Tokyo"
```

Tokenized sequence (simplified):

```
[Q, :, Japan, ?, A, :, Tokyo]
```

- Input IDs: `[Q, :, Japan, ?, A, :, Tokyo]`
- Labels: `[ -100, -100, -100, -100, -100, -100, Tokyo ]`

---

## ⚖️ Why this matters
- **Without masking**: The model learns to “parrot” the prompt, which is useless.  
- **With masking**: The model learns the mapping from prompt → response, which is exactly what you want in instruction‑tuning.

---

👉 So the **logic** is: `-100` marks tokens that should be ignored by the loss function, ensuring the model only learns to generate the *response* portion of your dataset.

---
Exactly — you’ve got it. Let me unpack the mechanics clearly:

---

## 🔎 How prediction works in a transformer LM
- At the **final layer**, the transformer produces a hidden state for **every position in the input sequence** (the whole context window).
- Each hidden state is passed through a **linear layer + softmax** over the vocabulary.  
  → That means you get a probability distribution over the entire vocab **for every token position**.

---

## ✅ Training vs Inference
- **Training**:  
  - You compute loss at *all positions* where labels are not `-100`.  
  - So the model is simultaneously learning to predict the next token at every step in the sequence.  
  - Example:  
    ```
    Input: [Q, :, Japan, ?, A, :, Tokyo]
    Labels: [-100, -100, -100, -100, -100, -100, Tokyo]
    ```
    → Loss is only applied at the “Tokyo” position.  
    → But technically the model produced logits for *all* positions.
- **Inference (generate)**:  
  - You only care about the **last position’s logits** (the next token prediction).  
  - You sample/argmax from that distribution, append the token, and repeat.

---

## ⚖️ Why mask with `-100`
- Because the model always outputs predictions for the **whole context window**, you need to tell the loss function which positions matter.  
- Masking prompt tokens with `-100` ensures the loss is only computed on the **response portion**.  
- Otherwise, the model would waste effort trying to “predict” the prompt tokens it already saw.

---

## 🧪 Analogy
Think of it like a classroom test:
- The teacher gives you the whole exam sheet (prompt + answer space).  
- You’re graded only on the **answer section**.  
- The prompt is context, not something you’re supposed to “predict back.”

---
## ***Adapter Composition***

When you train **different LoRA adapters** for different tasks (say summarization, translation, reasoning, coding), you don’t need to retrain the whole base model each time. Instead, you keep the base model frozen and **swap in/out the LoRA adapters depending on the task**.  

---

- In Hugging Face / PEFT terminology, this is often referred to as **adapter composition** or **LoRA adapter swapping**.  
- More generally, it’s part of **parameter‑efficient fine‑tuning (PEFT)** strategies.  
- Some frameworks call it **multi‑adapter inference** or **adapter fusion** when you combine multiple adapters at once.  
- In practice, people just say *“load the LoRA adapter for task X”*.

---

## ✅ How it works in code
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load adapter for summarization
summarization_model = PeftModel.from_pretrained(base_model, "summarization-lora")

# Later, swap to translation adapter
translation_model = PeftModel.from_pretrained(base_model, "translation-lora")
```

You can keep multiple adapters saved separately and load them depending on your config or task.

---

## ⚖️ Benefits
- **Efficiency**: Base model is reused, only small adapter weights change.  
- **Flexibility**: You can maintain a library of task‑specific adapters.  
- **Scalability**: Easy to deploy — just swap adapters at runtime.  

---

---
* ***Most of the local minima' are saddle point in high dimention space***
### 📌 Local Minima in LLM Training
- **Hard to detect**: Local minima are subtle and not easy to spot in large language models.  
- **Early convergence warning**: If the model converges too quickly, be cautious — it may be stuck in a local minimum.  
- **Validate results**: Always test thoroughly before accepting early convergence as “good enough.”

---

### ⚖️ Strategy to Avoid Local Minima
- **Checkpoint before trouble**: Save a checkpoint ~100 steps before the point where early convergence usually occurs.  
- **Lower learning rate**: Resume training from that checkpoint with a drastically reduced learning rate.  
- **Train past the minima**: Continue training until you’re confident the model has moved beyond the problematic region.  
- **Raise learning rate again**: Once past it, increase the learning rate back to normal to continue effective training.  

---

### ✅ Best Practices
- **Keep old checkpoints**: Don’t overwrite — maintain the earlier checkpoint as a fallback.  
- **Save new checkpoints**: After adjusting and training past the local minimum, save another checkpoint for recovery.  
- **Iterative safety net**: This checkpoint strategy ensures you can roll back if things go wrong.  

---
