"""
Enhanced K-NRM Models — Three improvements over the original paper:

  GAP 1 → C-KNRM   : Contextualized embeddings via DistilBERT (replaces static Word2Vec)
  GAP 2 → LK-NRM   : Learnable kernel parameters mu & sigma (replaces fixed hyperparameters)
  GAP 3 → Conv-KNRM: Convolutional aggregation over query dimension (replaces log-sum)

Each enhancement is modular so you can mix and match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 1 — LEARNABLE KERNELS  (fixes GAP 2)
# ══════════════════════════════════════════════════════════════════════════════

class LearnableKernelPooling(nn.Module):
    """
    Kernel pooling where mu (center) and sigma (width) are TRAINABLE parameters.

    Original: kernels are fixed, manually tuned to sigma ∈ [0.05, 0.2]
    Enhanced: backpropagation adjusts kernels automatically per dataset
    """

    def __init__(self, num_kernels=11):
        super().__init__()
        self.num_kernels = num_kernels

        # Initialize mus evenly in [-1, 1]; sigmas start at 0.1
        init_mus    = torch.linspace(-1.0, 1.0, num_kernels)
        init_sigmas = torch.full((num_kernels,), 0.1)
        # Last kernel = exact match
        init_mus[-1]    = 1.0
        init_sigmas[-1] = 0.001

        # These are now LEARNABLE (nn.Parameter gets updated by optimizer)
        self.mus    = nn.Parameter(init_mus)
        self.sigmas = nn.Parameter(init_sigmas)

    def forward(self, M, query_mask, doc_mask):
        """
        Args:
            M          : (B, Q, L)  translation matrix (cosine similarities)
            query_mask : (B, Q)     1 for real tokens, 0 for padding
            doc_mask   : (B, L)     1 for real tokens, 0 for padding

        Returns:
            phi : (B, Q, K) log soft-TF features
        """
        # Clamp sigma to stay positive and avoid numerical issues
        sigmas = torch.clamp(self.sigmas, min=1e-4)

        M_exp  = M.unsqueeze(-1)                          # (B, Q, L, 1)
        mus    = self.mus.view(1, 1, 1, -1)
        sigs   = sigmas.view(1, 1, 1, -1)

        K = torch.exp(-0.5 * ((M_exp - mus) / sigs) ** 2)   # (B, Q, L, K)

        doc_mask_exp = doc_mask.float().unsqueeze(1).unsqueeze(-1)  # (B,1,L,1)
        K = K * doc_mask_exp

        K_sum = K.sum(dim=2)                              # (B, Q, K)
        K_log = torch.log1p(K_sum)                        # (B, Q, K)

        q_mask = query_mask.float().unsqueeze(-1)         # (B, Q, 1)
        K_log  = K_log * q_mask

        return K_log   # (B, Q, K)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 2 — CONVOLUTIONAL AGGREGATION  (fixes GAP 3)
# ══════════════════════════════════════════════════════════════════════════════

class ConvAggregation(nn.Module):
    """
    Replaces the simple summation-over-query with a 1D CNN + MaxPool.

    Original: phi = sum_i log(1 + sum_j K(M_ij))   — ORDER INVARIANT (bag-of-words)
    Enhanced: slide a CNN window over query positions → captures n-gram patterns

    For example, the query "New York City" should match documents that contain
    those three words together, not scattered randomly. CNN detects adjacency.
    """

    def __init__(self, num_kernels=11, hidden_dim=64, kernel_sizes=(1, 2, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=num_kernels,
                      out_channels=hidden_dim,
                      kernel_size=k,
                      padding=k // 2)
            for k in kernel_sizes
        ])
        self.out_dim = hidden_dim * len(kernel_sizes)

    def forward(self, phi):
        """
        Args:
            phi : (B, Q, K) kernel features per query term

        Returns:
            agg : (B, out_dim) aggregated features
        """
        # Conv1d expects (B, channels, length) → (B, K, Q)
        x = phi.transpose(1, 2)          # (B, K, Q)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))          # (B, hidden, Q)
            h = h.max(dim=-1).values     # (B, hidden)  global max pool
            pooled.append(h)

        return torch.cat(pooled, dim=-1)  # (B, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED MODEL 1 — LK-NRM: Learnable Kernel + Conv (word-embedding based)
# ══════════════════════════════════════════════════════════════════════════════

class LKNRM(nn.Module):
    """
    LK-NRM: Original word-embedding backbone + Learnable Kernels + Conv Aggregation.
    Addresses GAP 2 and GAP 3 while keeping the same vocabulary-based embeddings.
    """

    def __init__(self, vocab_size, embed_dim=128, num_kernels=11,
                 pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings))

        self.kernel_pool = LearnableKernelPooling(num_kernels)
        self.conv_agg    = ConvAggregation(num_kernels)
        self.fc          = nn.Linear(self.conv_agg.out_dim, 1, bias=False)

    def forward(self, query_ids, doc_ids):
        q_emb = self.embedding(query_ids)
        d_emb = self.embedding(doc_ids)

        q_norm = F.normalize(q_emb, p=2, dim=-1)
        d_norm = F.normalize(d_emb, p=2, dim=-1)
        M = torch.bmm(q_norm, d_norm.transpose(1, 2))   # (B, Q, L)

        q_mask = (query_ids != 0)
        d_mask = (doc_ids   != 0)

        phi  = self.kernel_pool(M, q_mask, d_mask)     # (B, Q, K)
        feat = self.conv_agg(phi)                       # (B, out_dim)
        return self.fc(feat).squeeze(-1)                # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED MODEL 2 — C-KNRM: Contextualized (BERT) + Learnable Kernel + Conv
# ══════════════════════════════════════════════════════════════════════════════

class CKNRM(nn.Module):
    """
    C-KNRM: Full enhanced model addressing ALL THREE GAPS.

    GAP 1 → DistilBERT produces context-aware token vectors.
             "bank" in "river bank" ≠ "bank" in "bank deposit"
    GAP 2 → Learnable kernel mu and sigma via nn.Parameter
    GAP 3 → CNN aggregation over query positions detects phrase patterns

    Note: We use DistilBERT (smaller/faster BERT) for feasibility on student hardware.
    The original paper used Word2Vec on a 35M Sogou query log (proprietary).
    We use the publicly available MS MARCO dataset instead.
    """

    def __init__(self, num_kernels=11, bert_model='distilbert-base-uncased',
                 freeze_bert=True):
        super().__init__()

        # GAP 1 FIX: Pretrained Transformer encoder
        self.bert = DistilBertModel.from_pretrained(bert_model)
        self.embed_dim = self.bert.config.dim   # 768 for distilbert-base

        if freeze_bert:
            # Freeze most BERT layers, only fine-tune last 2 layers to save compute
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze last transformer layer
            for param in self.bert.transformer.layer[-1].parameters():
                param.requires_grad = True

        # Project BERT 768-dim to smaller dim to reduce computation of M matrix
        self.proj = nn.Linear(self.embed_dim, 128)

        # GAP 2 FIX: Learnable kernels
        self.kernel_pool = LearnableKernelPooling(num_kernels)

        # GAP 3 FIX: Convolutional aggregation
        self.conv_agg = ConvAggregation(num_kernels)

        self.fc = nn.Linear(self.conv_agg.out_dim, 1, bias=False)

    def encode(self, input_ids, attention_mask):
        """Get contextual token embeddings from BERT."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (B, seq_len, 768)
        h = outputs.last_hidden_state
        h = self.proj(h)                       # (B, seq_len, 128)
        return h

    def forward(self, q_ids, q_mask, d_ids, d_mask):
        """
        Args:
            q_ids, q_mask : (B, Q) query input_ids and attention_mask
            d_ids, d_mask : (B, L) document input_ids and attention_mask
        """
        q_emb = self.encode(q_ids, q_mask)    # (B, Q, 128)
        d_emb = self.encode(d_ids, d_mask)    # (B, L, 128)

        q_norm = F.normalize(q_emb, p=2, dim=-1)
        d_norm = F.normalize(d_emb, p=2, dim=-1)
        M = torch.bmm(q_norm, d_norm.transpose(1, 2))   # (B, Q, L)

        q_bool_mask = q_mask.bool()
        d_bool_mask = d_mask.bool()

        phi  = self.kernel_pool(M, q_bool_mask, d_bool_mask)   # (B, Q, K)
        feat = self.conv_agg(phi)                               # (B, out_dim)
        return self.fc(feat).squeeze(-1)                        # (B,)
