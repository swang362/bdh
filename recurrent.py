# Copyright Pathway Technology, Inc.

"""Recurrent (token-by-token) inference for trained BDH models.

BDH's attention has no softmax: position t reads

    y_t = sum_{s<t} (q_t . k_s) v_s = q_t @ S,   with   S = sum_{s<t} k_s^T v_s

so instead of re-reading the whole context for every new token, a fixed-size
state S (n_head x N x n_embd per layer, the "synapse" matrix of the paper) is
updated with one outer product per token. The weights are those of a normal
checkpoint; nothing is retrained.

Sliding window (window=W, the training block size by default): the state holds
at most W - 1 earlier tokens, the most a position sees in training. The oldest
token's contribution is subtracted when it leaves. While prompt + output fit in
W tokens, the result equals bdh.BDH.forward exactly (up to rounding). Beyond
that, each token's entry keeps the representation computed when it arrived
(like a transformer's sliding-window KV cache), rather than being recomputed
from a truncated window as inference.py's default method does.

window=None: unlimited. Nothing is ever removed, and memory stays constant, but
quality degrades past the training length because the state's sums grow larger
than anything seen in training.
"""

import collections

import torch
import torch.nn.functional as F

from bdh import Attention


def sample_next(logits, temperature=1.0, top_k=None):
    """Sample one token id from logits of shape (1, vocab); same rule as BDH.generate."""
    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class RecurrentBDH:
    """Runs a trained bdh.BDH model one token at a time with a fixed-size state."""

    def __init__(self, model, window=None):
        if window is not None and window < 1:
            raise ValueError("window must be at least 1, or None for unlimited")
        self.model = model
        cfg = model.config
        self.n_layer = cfg.n_layer
        self.nh = cfg.n_head
        self.D = cfg.n_embd
        self.N = cfg.mlp_internal_dim_multiplier * self.D // self.nh
        # a position attends to at most window - 1 earlier tokens (the diagonal is excluded)
        self.capacity = None if window is None else window - 1
        # RoPE frequencies; phases are computed in float64 on the CPU, so absolute
        # positions stay precise in very long streams (and on devices without float64)
        self.freqs = model.attn.freqs.detach().view(-1).double().cpu()
        self.device = model.lm_head.device
        self.reset()

    def reset(self):
        """Forget everything: empty state, position 0."""
        # the state is kept in float32: it is a long running sum
        self.S = [
            torch.zeros(self.nh, self.N, self.D, device=self.device, dtype=torch.float32)
            for _ in range(self.n_layer)
        ]
        # (rotated key, value) of each token still in the window, to subtract it when it leaves
        self.entries = [collections.deque() for _ in range(self.n_layer)]
        self.pos = 0
        self.ids = []

    def state_bytes(self):
        """Memory used by the state and the window buffers."""
        s = sum(t.numel() * t.element_size() for t in self.S)
        e = sum(q.numel() * q.element_size() + v.numel() * v.element_size() for d in self.entries for q, v in d)
        return s + e

    def _attend(self, layer, q, v):
        """Read the state for rotated query q (nh, N), then add this token (q, v)."""
        S = self.S[layer]
        y = torch.bmm(q.unsqueeze(1), S).squeeze(1)  # (nh, D): tokens before this one
        S.addcmul_(q.unsqueeze(-1), v.view(1, 1, -1))
        if self.capacity is not None:
            entries = self.entries[layer]
            entries.append((q, v))
            if len(entries) > self.capacity:
                old_q, old_v = entries.popleft()
                S.addcmul_(old_q.unsqueeze(-1), old_v.view(1, 1, -1), value=-1)
        return y

    @torch.no_grad()
    def step(self, token):
        """Feed one token; returns the logits (vocab,) for the next token."""
        m = self.model
        phases = ((self.pos * self.freqs) % 1).float().to(self.device)
        x = m.ln(m.embed.weight[token])  # (D,)
        for layer in range(self.n_layer):
            # same computation as BDH.forward for a single position
            x_sparse = F.relu(torch.einsum("d,hdn->hn", x, m.encoder))  # (nh, N)
            with torch.autocast(device_type=self.device.type, enabled=False):
                q = Attention.rope(phases, x_sparse.float())
                y_kv = self._attend(layer, q, x.float())
            y_kv = m.ln(y_kv)
            y_sparse = F.relu(torch.einsum("hd,hdn->hn", y_kv, m.encoder_v))
            xy_sparse = x_sparse * y_sparse
            y_mlp = xy_sparse.reshape(-1) @ m.decoder  # heads concatenated, as in forward
            x = m.ln(x + m.ln(y_mlp))
        self.pos += 1
        self.ids.append(token)
        return x @ m.lm_head

    @torch.no_grad()
    def prefill(self, ids):
        """Reset, then feed a prompt; returns the logits for the token after it.

        Up to one window of tokens is processed in parallel (one forward pass, the
        same computation as training); any further tokens are fed one at a time.
        """
        self.reset()
        n_parallel = len(ids) if self.capacity is None else min(len(ids), self.capacity + 1)
        logits = self._prefill_parallel(ids[:n_parallel]) if n_parallel else None
        for token in ids[n_parallel:]:
            logits = self.step(token)
        return logits

    def _prefill_parallel(self, ids):
        m = self.model
        T = len(ids)
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        keep = T if self.capacity is None else min(T, self.capacity)
        positions = torch.arange(T, device=self.device, dtype=torch.float32).view(1, 1, -1, 1)
        x = m.ln(m.embed(idx).unsqueeze(1))  # (1, 1, T, D)
        for layer in range(self.n_layer):
            x_sparse = F.relu(x @ m.encoder)  # (1, nh, T, N)
            y_kv = m.attn(Q=x_sparse, K=x_sparse, V=x)  # exactly as in forward
            with torch.autocast(device_type=self.device.type, enabled=False):
                # build the state from the last `keep` tokens' rotated keys and values
                q = Attention.rope(positions * m.attn.freqs, x_sparse.float())[0]  # (nh, T, N)
                v = x[0, 0].float()  # (T, D)
                self.S[layer] = torch.matmul(q[:, T - keep :].transpose(1, 2), v[T - keep :])
                if self.capacity is not None:
                    self.entries[layer].extend((q[:, t], v[t]) for t in range(T - keep, T))
            y_kv = m.ln(y_kv)
            y_sparse = F.relu(y_kv @ m.encoder_v)
            xy_sparse = x_sparse * y_sparse
            y_mlp = xy_sparse.transpose(1, 2).reshape(1, 1, T, -1) @ m.decoder
            x = m.ln(x + m.ln(y_mlp))
        self.pos = T
        self.ids = list(ids)
        return (x[0, 0, -1] @ m.lm_head)


@torch.no_grad()
def generate_stream_recurrent(model, ids, max_new_tokens, temperature=1.0, top_k=None, context_size=None, state=None):
    """Like inference.generate_stream, but recurrent. ids is a list of prompt token ids.

    context_size is the sliding window (None: unlimited). Pass a RecurrentBDH as
    state to reuse its buffers; it is reset first.
    """
    rec = state if state is not None else RecurrentBDH(model, window=context_size)
    logits = rec.prefill(ids)
    for i in range(max_new_tokens):
        token = sample_next(logits.float().unsqueeze(0), temperature, top_k).item()
        yield token
        if i + 1 < max_new_tokens:
            logits = rec.step(token)
