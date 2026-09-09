# K-NRM Enhanced — End-to-End Neural Ad-hoc Ranking with Kernel Pooling

**Team Quad Core** | ML Project Final Submission

---

## What This Project Does

This project **replicates** the K-NRM paper ([Xiong et al., 2017](https://arxiv.org/abs/1706.06613)) and then **enhances** it by addressing three specific limitations identified in the paper.

---

## Project Structure

```
knrm_project/
├── main.py                    ← Run this to train all models
├── demo.py                    ← Interactive ranking demo
├── requirements.txt
├── models/
│   ├── knrm.py               ← Original K-NRM (paper replication)
│   └── enhanced_knrm.py      ← Three enhanced models
├── utils/
│   ├── dataset.py            ← Data generation & loading
│   └── trainer.py            ← Training loop, loss, metrics
└── results/                  ← Saved weights + plots appear here
```

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train all models

```bash
# Train all 3 models (K-NRM + LK-NRM + C-KNRM)
python main.py

# Train only base K-NRM (original paper)
python main.py --model knrm

# Train only LK-NRM (learnable kernels + conv)
python main.py --model lknrm

# Train only C-KNRM (BERT-based, needs internet first time)
python main.py --model cknrm

# Skip BERT model (if no internet or slow machine)
python main.py --no-bert

# Custom epochs / batch size
python main.py --epochs 15 --batch-size 64
```

### Step 3 — Run demo (interactive ranking)

```bash
python demo.py --model knrm
python demo.py --model lknrm
python demo.py --model cknrm
```

---

## Windows Commands

```cmd
cd knrm_project
pip install -r requirements.txt
python main.py
python demo.py --model lknrm
```

## Linux/Mac Commands

```bash
cd knrm_project
pip install -r requirements.txt
python main.py
python demo.py --model lknrm
```

> **Note:** First time running C-KNRM will download DistilBERT (~250MB). This is cached locally after the first run.

---

## The Original Paper — K-NRM

### Problem
Traditional IR models like **BM25** require **exact word matches**. If you search "car", documents with "automobile" won't be found. While word embeddings (Word2Vec) help, they are too noisy — "Pittsburgh" and "Boston" appear similar because both are cities, leading to irrelevant results.

### K-NRM Solution

1. **Translation Matrix M** — Computes cosine similarity between every query word `q_i` and document word `d_j`. Shape: `(|Q| × |D|)`.

2. **Kernel Pooling** — Uses 11 RBF (Gaussian) kernels to "softly count" word pairs at different similarity levels:
   - Kernel 11: exact match (μ=1.0, σ=0.001)
   - Kernels 1-10: soft matches (μ evenly spaced in [-1,1], σ=0.1)
   
   Formula: `K_k(q_i) = log(1 + Σ_j exp(-(M_ij - μ_k)² / 2σ_k²))`

3. **Learning-to-Rank** — A single linear layer combines the K kernel scores into a final ranking score.

4. **Loss** — Pairwise margin loss: `L = max(0, 1 - score(q,pos) + score(q,neg))`

---

## The Three Gaps & Our Enhancements

### GAP 1 — Context Blindness (Static Embeddings)

**Problem:** Word2Vec gives "bank" the same vector in "river bank" and "bank deposit". The model struggles on rare ("tail") queries because Word2Vec needs many training examples per word.

**Our Fix — C-KNRM:** Replace Word2Vec with **DistilBERT**, a pretrained Transformer. BERT reads the full sentence, so the embedding for "bank" changes based on context. This is called **transfer learning** — we get semantic understanding without needing 35 million search logs.

```python
# Old: static lookup
q_emb = self.embedding(query_ids)   # same vector always

# New: context-aware
outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
q_emb = outputs.last_hidden_state   # different vector per context
```

---

### GAP 2 — Rigid Hyperparameters (Fixed Kernels)

**Problem:** The paper hard-codes kernel means (μ) and widths (σ). Authors had to manually search σ ∈ [0.05, 0.2] to avoid "too sharp" or "too flat" kernels. This doesn't generalize across datasets.

**Our Fix — Learnable Kernels:** Define μ and σ as **`nn.Parameter`** — they get updated by the optimizer via backpropagation, automatically finding the best similarity resolution.

```python
# Old: fixed buffer (not trained)
self.register_buffer('mus', torch.FloatTensor(mus))

# New: learnable parameter (updated by optimizer)
self.mus    = nn.Parameter(torch.linspace(-1.0, 1.0, num_kernels))
self.sigmas = nn.Parameter(torch.full((num_kernels,), 0.1))
```

---

### GAP 3 — Structure Loss (Bag-of-Words Aggregation)

**Problem:** The paper uses `Σ_i log(1 + Σ_j K_k(M_ij))` — summing over all query positions. This is **order-invariant**: the model cannot distinguish "blind venetian" (a person's name) from "venetian blind" (window shade) because word order is lost.

**Our Fix — Conv-KNRM:** Replace the sum with a **1D Convolutional layer** followed by Max Pooling. The CNN slides a window over query positions and detects **n-gram patterns** where adjacent high-relevance matches occur together.

```python
# Old: simple sum (loses word order)
phi = K_log.sum(dim=1)   # (B, K)

# New: CNN captures local phrase structure
x = phi.transpose(1, 2)                # (B, K, Q)
h = F.relu(conv(x))                    # (B, hidden, Q)
feat = h.max(dim=-1).values            # (B, hidden) — global max pool
```

---

## Model Comparison

| Model | Embedding | Kernel Type | Aggregation | Fixes |
|-------|-----------|-------------|-------------|-------|
| **K-NRM** | Word2Vec (static) | Fixed μ, σ | Log-sum | Baseline |
| **LK-NRM** | Word2Vec (static) | Learnable μ, σ | CNN + MaxPool | GAP 2, GAP 3 |
| **C-KNRM** | DistilBERT (contextual) | Learnable μ, σ | CNN + MaxPool | GAP 1, GAP 2, GAP 3 |

---

## Metrics Used

- **Pairwise Accuracy** — What fraction of (pos, neg) pairs does the model rank correctly? (Higher is better)
- **NDCG@5** — Normalized Discounted Cumulative Gain at rank 5. Standard IR metric. (Higher is better, max = 1.0)
- **Training Loss** — Pairwise margin loss (Lower is better)

The original paper reports **NDCG@1, NDCG@10, and MRR** on the Sogou dataset. We use the same metrics adapted to our synthetic dataset which mirrors the pairwise training structure.

---

## Dataset Note

The original paper uses the **proprietary Sogou.com search log** (35M sessions). Since this is not publicly available, we use a **synthetic dataset** that mirrors the same pairwise training structure:
- Each training example: `(query, relevant_doc, non-relevant_doc)`
- Queries and documents are from 8 topic clusters covering IR, ML, NLP topics
- Hard negatives: relevant documents from *other* topics (forces the model to learn fine-grained distinctions)

This is the standard approach when the original dataset is unavailable — the same architecture and training procedure is used, just on accessible data.

---

## Key Takeaways (For Presentation)

1. **K-NRM's key insight**: Instead of binary word matching, use Gaussian kernels to create a *spectrum* of match strengths — from exact to loosely related.

2. **Why kernels work**: The soft-TF features capture *how often* query terms match document terms at *each similarity level* — much richer than just counting exact matches.

3. **Why our enhancements work**:
   - BERT fixes the "word meaning" problem
   - Learnable kernels fix the "one-size-fits-all" sigma problem
   - CNN fixes the "word order ignored" problem

4. **End-to-end training** is the key differentiator from classical IR: all components (embeddings, kernels, ranking layer) are jointly optimized toward ranking accuracy.
