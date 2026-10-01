"""A Llama-style GPT, as a baseline for comparing with BDH under the same data,
tokenizer and training code (train.py --arch gpt).

Pre-norm transformer: RMSNorm, rotary position embeddings (RoPE), causal
softmax attention (PyTorch's fused scaled_dot_product_attention), a SwiGLU MLP,
and input embeddings tied to the output layer. No biases.

Same interface as bdh.BDH: forward(idx, targets=None) -> (logits, loss).
"""

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class GPTConfig:
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    dropout: float = 0.0
    vocab_size: int = 256
    mlp_hidden: int = 0  # SwiGLU hidden size; 0 means 8/3 * n_embd, rounded up to a multiple of 64
    rope_theta: float = 10000.0

    def hidden_size(self):
        return self.mlp_hidden or -(-8 * self.n_embd // 3 // 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def rope_cos_sin(T, head_dim, theta, device):
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(T, device=device).float(), freqs)  # (T, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, nh, T, head_dim); rotates pairs (x1[i], x2[i]) from the two halves
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).type_as(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, H = config.n_embd, config.hidden_size()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.attn_norm = RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.mlp_norm = RMSNorm(D)
        self.gate = nn.Linear(D, H, bias=False)
        self.up = nn.Linear(D, H, bias=False)
        self.down = nn.Linear(H, D, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, D = x.size()
        q, k, v = self.qkv(self.attn_norm(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        h = self.mlp_norm(x)
        return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0 and (config.n_embd // config.n_head) % 2 == 0
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        # GPT-2-style init: normal(0.02), with the projections that add into the
        # residual stream scaled down by depth so its variance stays bounded
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / (2 * config.n_layer) ** 0.5 if name.endswith(("proj.weight", "down.weight")) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        C = self.config
        cos, sin = rope_cos_sin(T, C.n_embd // C.n_head, C.rope_theta, idx.device)
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.norm(x) @ self.embed.weight.T  # output layer tied to the embeddings
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # same as bdh.BDH.generate, used for train.py's sample after training
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
