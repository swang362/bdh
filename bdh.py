# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class ChunkedAttention(torch.autograd.Function):
    """(Q @ Q.mT).tril(-1) @ V, computed chunk by chunk with a running state.

    Within a chunk, attention is computed directly; earlier chunks contribute
    through the state S = sum of k^T v over all earlier tokens (B, nh, N, D),
    as in recurrent.py. Cost grows linearly with T instead of quadratically, and
    no T x T matrix is formed. The backward pass walks the chunks in reverse and
    rebuilds each earlier state by subtracting the chunk's k^T v from the final
    state, so only one state is kept in memory, not one per chunk.

    Q: (B, nh, T, N) rotated sparse activations (also the keys); V: (B, 1, T, D).
    Matmuls run in Q's dtype (bf16 under autocast); the state is summed in at
    least float32.
    """

    @staticmethod
    def forward(ctx, Q, V, chunk):
        with torch.autocast(Q.device.type, enabled=False):
            ctx.v_dtype = V.dtype
            V = V.to(Q.dtype)
            B, nh, T, N = Q.shape
            out = Q.new_empty(B, nh, T, V.size(-1))
            S = Q.new_zeros(B, nh, N, V.size(-1), dtype=torch.promote_types(Q.dtype, torch.float32))
            for s in range(0, T, chunk):
                q, v = Q[:, :, s : s + chunk], V[:, :, s : s + chunk]
                o = (q @ q.mT).tril(diagonal=-1) @ v
                if s:
                    o += q @ S.to(q.dtype)
                out[:, :, s : s + chunk] = o
                S += q.mT @ v
        ctx.save_for_backward(Q, V, S)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, V, S = ctx.saved_tensors
        chunk = ctx.chunk
        with torch.autocast(Q.device.type, enabled=False):
            dO = dO.to(Q.dtype)
            T = Q.size(2)
            dQ = torch.empty_like(Q)
            dV = torch.empty_like(V)
            P = S.clone()  # state before the current chunk
            G = torch.zeros_like(S)  # sum of q^T dO over later chunks
            for s in reversed(range(0, T, chunk)):
                q, v, do = Q[:, :, s : s + chunk], V[:, :, s : s + chunk], dO[:, :, s : s + chunk]
                P -= q.mT @ v
                # inside the chunk: o = A v, with A = tril(q q^T); q is both query and key
                A = (q @ q.mT).tril(diagonal=-1)
                dA = (do @ v.mT).tril(diagonal=-1)
                dq = dA @ q + dA.mT @ q
                dv = A.mT @ do
                # earlier chunks: o += q P; this chunk's k^T v reaches later chunks through G
                Gq = G.to(q.dtype)
                if s:
                    dq += do @ P.to(q.dtype).mT
                dq += v @ Gq.mT
                dv += q @ Gq
                dQ[:, :, s : s + chunk] = dq
                dV[:, :, s : s + chunk] = dv.sum(1, keepdim=True)  # V is shared by all heads
                G += q.mT @ do
        return dQ, dV.to(ctx.v_dtype), None


@torch.compiler.disable
def chunked_attention(Q, V, chunk):
    return ChunkedAttention.apply(Q, V, chunk)


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )
        # tokens per chunk for ChunkedAttention; None computes the full T x T scores.
        # Not part of the config: it changes how attention is computed, not the model.
        self.chunk_size = None

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        if self.chunk_size and T > self.chunk_size:
            return chunked_attention(QR, V, self.chunk_size)

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
