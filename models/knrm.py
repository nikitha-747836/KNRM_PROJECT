"""
K-NRM: End-to-End Neural Ad-hoc Ranking with Kernel Pooling
Paper: https://arxiv.org/abs/1706.06613

This is the BASE model — faithful replication of the original paper's approach.
Uses static Word2Vec-style embeddings + fixed RBF kernels + log-sum aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KNRM(nn.Module):
    """
    Original K-NRM model from the paper.

    Architecture:
      1. Embedding Layer  -> word vectors (static Word2Vec style)
      2. Translation Matrix M -> cosine similarity between every query word and document word
      3. Kernel Pooling   -> 11 RBF kernels (1 exact + 10 soft) extract soft-TF features
      4. Learning-to-Rank -> linear layer combines kernel outputs into final score

    Args:
        vocab_size  : number of words in vocabulary
        embed_dim   : word embedding dimension (paper uses 300)
        num_kernels : total number of kernels (paper uses 11)
        sigma       : width of soft kernels (paper uses 0.1)
        pretrained_embeddings : optional numpy array of shape (vocab_size, embed_dim)
    """

    def __init__(self, vocab_size, embed_dim=300, num_kernels=11, sigma=0.1,
                 pretrained_embeddings=None):
        super(KNRM, self).__init__()

        # ── 1. Word Embedding Layer ──────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings))

        # ── 2. Kernel Parameters (FIXED in base model) ───────────────────────
        # Exact-match kernel: mu=1.0, sigma very small
        # Soft kernels: mu evenly spaced from -1 to 1
        self.num_kernels = num_kernels
        self.sigma = sigma

        # Kernel means: 1 exact match (mu=1) + (num_kernels-1) soft kernels
        # spaced from -1+1/(num_kernels-1) to 1-1/(num_kernels-1)
        mus = torch.FloatTensor(self._kernel_mus(num_kernels))   # shape (K,)
        sigmas = torch.FloatTensor(self._kernel_sigmas(num_kernels, sigma))  # shape (K,)

        # Register as buffers (not trained in base model)
        self.register_buffer('mus', mus)
        self.register_buffer('sigmas', sigmas)

        # ── 3. Learning-to-Rank Layer ─────────────────────────────────────────
        # Input: num_kernels features per query term, then averaged over query
        # This linear layer assigns a weight to each kernel
        self.fc = nn.Linear(num_kernels, 1, bias=False)
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)

    def _kernel_mus(self, n_kernels):
        """Generate kernel means. Last kernel is exact-match at mu=1."""
        l_mu = [1.0]  # exact match kernel
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)          # span [-1, 1]
        l_mu = [1 - bin_size * (i + 1) for i in range(n_kernels - 1)] + [1.0]
        return l_mu

    def _kernel_sigmas(self, n_kernels, sigma):
        """All soft kernels share the same sigma; exact match uses tiny sigma."""
        l_sigma = [0.001] + [sigma] * (n_kernels - 1)
        return l_sigma

    def forward(self, query_ids, doc_ids):
        """
        Args:
            query_ids : (batch, q_len)   padded query token ids
            doc_ids   : (batch, d_len)   padded document token ids

        Returns:
            score     : (batch,)         ranking score
        """
        # 1. Embed
        q_emb = self.embedding(query_ids)     # (B, Q, D)
        d_emb = self.embedding(doc_ids)       # (B, L, D)

        # 2. Build Translation Matrix (cosine similarity)
        q_norm = F.normalize(q_emb, p=2, dim=-1)   # (B, Q, D)
        d_norm = F.normalize(d_emb, p=2, dim=-1)   # (B, L, D)
        # M[b, i, j] = cosine sim between query word i and doc word j
        M = torch.bmm(q_norm, d_norm.transpose(1, 2))   # (B, Q, L)

        # 3. Kernel Pooling
        # For each kernel k: K_k(M) = sum_j exp(-(M - mu_k)^2 / (2*sigma_k^2))
        # Expand M for broadcasting: (B, Q, L, 1)
        M_exp = M.unsqueeze(-1)                         # (B, Q, L, 1)
        mus   = self.mus.view(1, 1, 1, -1)              # (1, 1, 1, K)
        sigs  = self.sigmas.view(1, 1, 1, -1)           # (1, 1, 1, K)

        # Gaussian kernel values
        K = torch.exp(-0.5 * ((M_exp - mus) / sigs) ** 2)   # (B, Q, L, K)

        # Create padding mask: ignore positions where doc token is 0 (PAD)
        doc_mask = (doc_ids != 0).float().unsqueeze(1).unsqueeze(-1)  # (B,1,L,1)
        K = K * doc_mask

        # Sum over document dimension (soft-TF count per query term per kernel)
        K_sum = K.sum(dim=2)                            # (B, Q, K)

        # Log normalization (as in paper: log(1 + sum))
        K_log = torch.log1p(K_sum)                      # (B, Q, K)

        # Create query mask: ignore PAD query positions
        q_mask = (query_ids != 0).float().unsqueeze(-1)  # (B, Q, 1)
        K_log = K_log * q_mask

        # Sum over query dimension -> one feature vector per document
        phi = K_log.sum(dim=1)                          # (B, K)

        # 4. Learning-to-Rank
        score = self.fc(phi).squeeze(-1)                # (B,)
        return score
