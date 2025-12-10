# Knowledge Distillation

* **Knowledge Distillation allows a smaller model to learn from a foundation model to replicate similar behavior with fewer parameters. The student model does not always learn the emergent qualities of the foundation model, so the dataset must be especially curated. The dotted line is indicating a special relationship as the Student Model becomes the Specialized LLM.**

*  the student model is trained on the same task as the teacher model. However, instead of learning from the raw data directly, the student model learns to mimic the teacher model's outputs. This is typically done by adding a term to the loss function that encourages the student model's predictions to be similar to the teacher model's predictions. This means that the student model not only learns from the task-specific labels, but also benefits from the rich representations learned by the teacher model


---

## 1. What is this KL divergence part doing?

```python
loss_function = nn.KLDivLoss(reduction="batchmean")

loss_logits = loss_function(
    F.log_softmax(
        outputs_student.logits / self.args.temperature, dim=-1
    ),
    F.softmax(
        outputs_teacher.logits / self.args.temperature, dim=-1
    ),
) * (self.args.temperature ** 2)
```

### KLDivLoss

`nn.KLDivLoss` computes the Kullback–Leibler divergence between two probability distributions:

* Input: log probabilities (log p)
* Target: normal probabilities (q)

So we pass:

* Student: `F.log_softmax(student_logits / T)`  → log probabilities
* Teacher: `F.softmax(teacher_logits / T)`      → probabilities

`reduction="batchmean"` means:

* It sums the KL divergence over classes for each example
* Then averages over the batch

So `loss_logits` is the average KL divergence between student and teacher distributions across the batch.

### Temperature scaling

We do

```python
outputs_student.logits / self.args.temperature
outputs_teacher.logits / self.args.temperature
```

and then multiply by `T**2` at the end.

Why:

1. Dividing logits by a temperature > 1 makes the softmax output softer.

   * For example, probabilities become less extreme
   * This reveals more information about class similarities

2. Multiplying by `T**2` is a trick from Hinton et al so that the gradient scale is similar to what it would be without temperature. If you did not multiply by `T**2`, the gradients would shrink.

So conceptually:

* Teacher produces a soft target distribution over labels
* Student is trained to match that distribution

---

## 2. Why is the normal loss also there?

```python
loss = (
    self.args.alpha * student_loss
    + (1.0 - self.args.alpha) * loss_logits
)
```

Here you combine two things:

1. `student_loss`

   * This is the standard supervised loss (cross entropy) with real labels.
   * It trains the student to predict the correct class.

2. `loss_logits`

   * This is the distillation loss (KL), which trains the student to match the teacher distribution.

`alpha` controls the tradeoff:

* If `alpha = 1.0`
  Only ground truth loss, no distillation

* If `alpha = 0.0`
  Only distillation, ignore real labels

In practice, people use something like 0.5.
Intuition:

* Ground truth labels keep the student anchored to the actual task
* Teacher soft targets transfer the teacher knowledge, including information about class similarities

---

## 3. How is `student_loss` computed without labels being passed here?

You are right that inside `compute_loss` you never write `labels=...`. But the labels are still there.

Key points:

1. You renamed `"label"` to `"labels"`:

   ```python
   tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
   ```

2. The Hugging Face `Trainer` builds batches that look like:

   ```python
   {
       "input_ids": ...,
       "attention_mask": ...,
       "labels": ...
   }
   ```

3. In your `compute_loss`, you do:

   ```python
   outputs_student = model(**inputs)
   ```

   `inputs` includes `"labels"`, so this is effectively:

   ```python
   outputs_student = model(
       input_ids=...,
       attention_mask=...,
       labels=...
   )
   ```

4. `AutoModelForSequenceClassification` has a forward method that, if `labels` are present, automatically computes the standard cross entropy loss and returns it as `outputs_student.loss`.

So the labels are indeed passed, just hidden inside `inputs` and used by the model internally.

That is why `student_loss = outputs_student.loss` works.

---

## 4. What does this line do?

```python
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
```

This does several things:

1. `from_pretrained(student_id)`

   * Loads the pretrained weights of `gpt2` (your student) plus a classification head on top.

2. `num_labels=num_labels`

   * Tells the model how many classes your task has.
   * For SST2 that is usually 2: "negative" and "positive".
   * This defines the size of `logits`: `[batch_size, num_labels]`.

3. `id2label=id2label` and `label2id=label2id`

   * These are dictionaries like:

     ```python
     id2label = {"0": "negative", "1": "positive"}
     label2id = {"negative": "0", "positive": "1"}
     ```

   * They are stored in `model.config`.

   * They do not affect training mathematically.

   * They are useful when you run pipelines or print predictions:

     * The model can say `"positive"` instead of just class index `1`.

So these arguments are mostly about:

* Adapting the classification head to the right number of labels
* Making the model outputs human readable

---

## 5. Summary in one shot

* `KLDivLoss` compares the student probability distribution with the teacher distribution.
* You soften both distributions with a temperature and rescale gradients with `T**2`.
* `student_loss` is the usual cross entropy with real labels, computed automatically because `labels` are inside `inputs` and HF models compute it when labels are present.
* Final loss is a mixture of ground truth loss and distillation loss, controlled by `alpha`.
* `num_labels`, `id2label`, `label2id` configure the classification head and label names, not the core training math.

### KLDivLoss in pytorch

---

# Why is there a log in KLDivLoss?

Why do we do:

```python
F.log_softmax(student_logits)
F.softmax(teacher_logits)
```

instead of giving both as normal softmax probabilities?

---

# 1. KL divergence formula from probability theory

Classical KL divergence is:

```
KL(p || q) = Σ p(x) * log( p(x) / q(x) )
```

This formula takes:

* `p(x)` normal probabilities
* `q(x)` normal probabilities
* Inside you compute `log(p(x)) - log(q(x))`

So **in theory** you give raw probabilities.

---

# 2. PyTorch’s KLDivLoss DOES NOT expect p and q directly

PyTorch changed the convention to make training more numerically stable.

PyTorch’s `nn.KLDivLoss` expects:

* **input**: log probability
* **target**: probability

This is written in the docs literally:

✔ input → log p(x)
✔ target → q(x)

So PyTorch computes:

```
KL(input || target) = Σ target * (log(target) - input)
```

That matches the classical formula when:

```
input = log p
target = q
```

---

# 3. That is why you must use:

### For the student (input → log probabilities):

```python
F.log_softmax(student_logits / T, dim=-1)
```

### For the teacher (target → probabilities):

```python
F.softmax(teacher_logits / T, dim=-1)
```

So the code is matching PyTorch’s requirement.

---

# 4. Why design it this way?

Two reasons.

### Reason 1. Numerical stability

`softmax` can produce values extremely close to zero, leading to:

* log(0)
* underflow
* NaN gradients

Computing log probabilities directly with `log_softmax` reduces overflow risks by combining:

```
log(exp(x - max(x))) = x - max(x)
```

### Reason 2. Backpropagation efficiency

Log softmax is more stable and gives better gradients.

---

# 5. If you gave softmax to both sides, it would be wrong

If you did:

```python
loss_function(
    F.softmax(student_logits),
    F.softmax(teacher_logits),
)
```

This breaks PyTorch’s contract:

* The input must be log probabilities
* The target must be probabilities

PyTorch would compute something completely different from KL divergence.

---

# 6. So the correct pattern is always:

```
KLDivLoss(
    input=log p,
    target=q
)
```

Which is exactly what your code does:

```python
loss_logits = loss_function(
    F.log_softmax(student_logits / T),
    F.softmax(teacher_logits / T),
)
```

---

# Summary

| What KL wants theoretically    | What PyTorch KLDivLoss requires            |
| ------------------------------ | ------------------------------------------ |
| p(x) and q(x) probabilities    | log p(x) as input, q(x) as target          |
| You compute log inside formula | PyTorch wants log on student ahead of time |
| KL = sum p log p/q             | KL = sum q (log q − log p)                 |

So the log is not duplicated.
It is required because PyTorch’s KL implementation expects log probabilities as the **input**.

---

---

# Let’s break it down clearly

## 1. What naive softmax then log would look like

If you literally did:

```python
p = softmax(x)
log_p = log(p)
```

that would be:

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)

log_softmax(x_i) = x_i - log(Σ exp(x_j))
```

But computing this directly is unstable because:

* `exp(x)` can overflow to infinity
* `softmax(x)` can become tiny values like 1e−50, causing `log` to blow up

---

# 2. PyTorch’s `log_softmax` uses a safer formula

PyTorch uses the numerically stable version:

```
log_softmax(x_i) = x_i - log(Σ exp(x_j))
```

This version avoids computing `exp(x_i) / sum(exp(x_j))` first.

And to improve stability, it also subtracts `max(x)`:

```
log_softmax(x_i)
= x_i - max(x) - log(Σ exp(x_j - max(x)))
```

This keeps all exponent values small enough to avoid overflow.

---

# 3. So the real sequence is:

### PyTorch `log_softmax` does:

1. Finds the maximum logit
2. Subtracts it from all logits
3. Computes exp of the shifted logits
4. Sums those exponentials
5. Takes log of that sum
6. Subtracts the log sum from each original shifted logit

It never calculates the softmax itself.

---

# 4. Why this matters for KL divergence?

`KLDivLoss` expects:

* input: log p
* target: q

So we supply:

```python
log_p = F.log_softmax(student_logits / T)
q     = F.softmax(teacher_logits / T)
```

If you used `softmax` then `log`, it would technically still work, but:

* It would be slower
* It could overflow
* It could produce NaNs

That is why `log_softmax` exists.

---

# 5. Summary

### Does `log_softmax` do softmax then log?

No. It computes **log softmax directly** in a safer way.

### Why do we use it?

Because:

* KLDivLoss requires log probabilities
* It avoids numerical instability
* It avoids computing softmax explicitly

---

You are thinking in the right direction.
Let me explain this clearly so you understand exactly what must match between teacher and student in distillation.

---

# Key idea

The student and teacher **must produce logits with the same shape** so the KL divergence makes sense.

```
student logits: [batch, num_labels]
teacher logits: [batch, num_labels]
```

If these shapes differ, KL loss cannot be computed.

---

# What needs to match depends on the task

## Case 1: Classification distillation

Your current code is **sequence classification**, not next token prediction.

For classification, the condition is simple:

### Only the number of labels must match

That is:

```python
num_labels = num_labels
```

Tokenizer vocabulary size does not matter because classification heads sit on top of the model and output a fixed number of logits equal to the number of classes.

Your setup:

* Teacher: GPT2 medium
* Student: GPT2
* Both loaded with

  ```python
  AutoModelForSequenceClassification(... num_labels=2)
  ```

So they output 2 logits each: positive vs negative.
Distillation works perfectly here.

Tokenizer does not need to be identical for classification.
But you used the same tokenizer anyway which is fine.

---

## Case 2: Language model distillation

This is a different setting.
If you were doing next token prediction, then:

### Student and teacher must share the same tokenizer and vocab size

Why:

* LM head outputs logits = vocabulary size
* If teachers vocab is a different size from student vocab
  output logits shape mismatch
  distillation breaks

For example:

* Teacher vocab size: 50257
* Student vocab size: 30000
  Shapes differ
  Cannot compute KL divergence

So for language model distillation, you must use the same tokenizer, or upconvert the smaller vocab model with `resize_token_embeddings`.

---

# In your case

You are doing **classification**, not next token prediction.

So:

* Tokenizers should match for convenience and consistent token alignment
* Vocab size does not affect KL divergence
* The only requirement is that both models have the same number of output labels

Thus your setup is correct.

---

# Short summary

### For classification distillation

Only `num_labels` must match.
Tokenizer can differ, but matching tokenizer avoids misalignment.

### For language model distillation

Vocabulary sizes must match.
Tokenizers must match.
Logit dimensions must match.

---

* ***Knowledge distillation, as exemplified by the provided compute_loss method, is a technique that enables the ransfer of valuable insights from a teacher model to a more lightweight student model. In this process, the teacher model provides soft targets, offering probability distributions over possible outputs, which are then utilized to train the student model. The critical aspect of knowledge distillation lies in the alignment of these distributions, ensuring that the student model not only learns to mimic the teacher's predictions but also gains a deeper understanding of the underlying data. This approach helps improve the student's generalization capabilities and performance on various tasks, ultimately making it more efficient and adaptable***