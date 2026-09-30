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
token's contribution is subtracted when it leaves; the last W - 1 tokens' keys
and values are kept in a ring buffer for that. While prompt + output fit in W
tokens, the result equals bdh.BDH.forward exactly (up to rounding). Beyond
that, each token's entry keeps the representation computed when it arrived
(like a transformer's sliding-window KV cache), rather than being recomputed
from a truncated window as inference.py's default method does.

window=None: unlimited. Nothing is ever removed, and memory stays constant, but
quality degrades past the training length because the state's sums grow larger
than anything seen in training.

CUDA graphs (cuda_graph=True): at batch size 1 a token step is ~100 small GPU
operations, and launching them from Python costs more than running them. The
whole step is recorded once as a CUDA graph and then replayed with one launch.
All state lives in fixed, preallocated tensors (zeroed in place by reset), so
the recorded graph stays valid across prompts and chat turns.

Saving and loading (save / load): the state can be written to a file and read
back later to continue exactly where it stopped. The file records a fingerprint
of the model (config and a checksum of the weights), so a state can't be loaded
into a different model. With the sliding window, the state holds only the last
window - 1 tokens: it's a session checkpoint, not a memory of a whole document.
"""

import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

from bdh import Attention

STATE_FORMAT = "bdh-recurrent-state-v1"


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

    def __init__(self, model, window=None, cuda_graph=False):
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
        self.device = model.lm_head.device
        freqs = model.attn.freqs.detach().view(-1).double()
        # RoPE phases are computed in float64, so absolute positions stay precise in
        # very long streams; on the device where supported, else on the CPU (e.g. MPS)
        self.phases_on_device = self.device.type != "mps"
        self.freqs = freqs.to(self.device) if self.phases_on_device else freqs.cpu()

        # all state is preallocated once and only ever modified in place (CUDA graphs
        # replay fixed memory addresses); float32, since S is a long running sum
        f32 = dict(device=self.device, dtype=torch.float32)
        self.S = [torch.zeros(self.nh, self.N, self.D, **f32) for _ in range(self.n_layer)]
        cap = self.capacity or 0
        self.ring_q = [torch.zeros(cap, self.nh, self.N, **f32) for _ in range(self.n_layer)] if cap else []
        self.ring_v = [torch.zeros(cap, self.D, **f32) for _ in range(self.n_layer)] if cap else []
        self.pos_t = torch.zeros(1, dtype=torch.long, device=self.device)  # position of the next token
        self.slot_t = torch.zeros(1, dtype=torch.long, device=self.device)  # ring slot of the oldest entry
        self.token_t = torch.zeros(1, dtype=torch.long, device=self.device)  # input of the next step
        self.pos = 0
        self.ids = []
        self.last_logits = None  # logits for the token after everything fed so far

        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        if cuda_graph and not self.cuda_graph:
            print("CUDA graphs need a CUDA device; running without them", file=sys.stderr)
        self.graph = None
        self.graph_logits = None

    def reset(self):
        """Forget everything: empty state, position 0."""
        for t in self.S + self.ring_q + self.ring_v:
            t.zero_()
        self.pos_t.zero_()
        self.slot_t.zero_()
        self.pos = 0
        self.ids = []
        self.last_logits = None

    @property
    def window(self):
        return None if self.capacity is None else self.capacity + 1

    def state_bytes(self):
        """Memory used by the state and the ring buffer."""
        return sum(t.numel() * t.element_size() for t in self.S + self.ring_q + self.ring_v)

    def _state_tensors(self):
        return self.S + self.ring_q + self.ring_v + [self.pos_t, self.slot_t]

    # ------------------------------------------------------------ snapshots and files

    def snapshot(self):
        """An in-memory copy of the state (on the same device), for restore()."""
        return {
            "tensors": [t.clone() for t in self._state_tensors()],
            "pos": self.pos,
            "ids": list(self.ids),
            "last_logits": None if self.last_logits is None else self.last_logits.clone(),
        }

    def restore(self, snap):
        """Go back to a snapshot(), copying in place (a recorded CUDA graph stays valid)."""
        for t, s in zip(self._state_tensors(), snap["tensors"]):
            t.copy_(s)
        self.pos = snap["pos"]
        self.ids = list(snap["ids"])
        self.last_logits = snap["last_logits"]

    def fingerprint(self):
        """Identifies the model a state belongs to: its config and a checksum of the weights."""
        m = self.model
        return {
            "config": dataclasses.asdict(m.config),
            "weights_checksum": [
                float(m.lm_head.detach().double().sum()),
                float(m.embed.weight.detach().double().abs().sum()),
                float(m.encoder.detach().double().abs().sum()),
            ],
        }

    def save(self, path, extra=None):
        """Write the state to path. extra: a dict of plain data (e.g. chat history) to store with it."""
        state = {
            "format": STATE_FORMAT,
            **self.fingerprint(),
            "window": self.window,
            "pos": self.pos,
            "slot": int(self.slot_t.item()),
            "ids": list(self.ids),
            "S": [t.cpu() for t in self.S],
            "ring_q": [t.cpu() for t in self.ring_q],
            "ring_v": [t.cpu() for t in self.ring_v],
            "last_logits": None if self.last_logits is None else self.last_logits.float().cpu(),
            "extra": extra or {},
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # write to a temp file first so an interrupted save never corrupts an existing state
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path):
        """Read a state written by save(); returns its extra dict.

        Raises ValueError if the file belongs to a different model or window size.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"{path} is not a BDH recurrent state file")
        mine = self.fingerprint()
        same_weights = all(
            abs(a - b) <= 1e-8 * max(1.0, abs(b)) for a, b in zip(state["weights_checksum"], mine["weights_checksum"])
        )
        if state["config"] != mine["config"] or not same_weights:
            raise ValueError(f"{path} was saved with a different model (config or weights differ)")
        if state["window"] != self.window:
            raise ValueError(
                f"{path} was saved with context window {state['window'] or 'unlimited'}, "
                f"but this run uses {self.window or 'unlimited'}; pass --context-size {state['window'] or 0}"
            )
        for t, s in zip(self.S + self.ring_q + self.ring_v, state["S"] + state["ring_q"] + state["ring_v"]):
            t.copy_(s)
        self.pos = state["pos"]
        self.pos_t.fill_(self.pos)
        self.slot_t.fill_(state["slot"])
        self.ids = list(state["ids"])
        last = state["last_logits"]
        self.last_logits = None if last is None else last.to(self.device)
        return state["extra"]

    def _phases(self):
        if self.phases_on_device:
            return ((self.pos_t.double() * self.freqs) % 1).float()
        return ((self.pos * self.freqs) % 1).float().to(self.device)

    def _attend(self, layer, q, v):
        """Read the state for rotated query q (nh, N), then add this token (q, v)."""
        S = self.S[layer]
        y = torch.bmm(q.unsqueeze(1), S).squeeze(1)  # (nh, D): tokens before this one
        if self.capacity == 0:
            return y  # window 1: no earlier token is ever kept
        S.addcmul_(q.unsqueeze(-1), v.view(1, 1, -1))
        if self.capacity is not None:
            # subtract the oldest entry (all zeros while the ring isn't full yet, which
            # is an exact no-op) and put this token in its slot
            ring_q, ring_v = self.ring_q[layer], self.ring_v[layer]
            old_q = ring_q.index_select(0, self.slot_t)[0]
            old_v = ring_v.index_select(0, self.slot_t)[0]
            S.addcmul_(old_q.unsqueeze(-1), old_v.view(1, 1, -1), value=-1)
            ring_q.index_copy_(0, self.slot_t, q.unsqueeze(0))
            ring_v.index_copy_(0, self.slot_t, v.unsqueeze(0))
        return y

    def _step_body(self):
        """One token step on tensors only (no Python-side decisions), so it can be graphed."""
        m = self.model
        phases = self._phases()
        x = m.ln(m.embed.weight.index_select(0, self.token_t)[0])  # (D,)
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
        self.pos_t.add_(1)
        if self.capacity:
            self.slot_t.add_(1).remainder_(self.capacity)
        return x @ m.lm_head

    def _capture(self):
        """Record _step_body as a CUDA graph; falls back to normal execution on failure."""
        try:
            # warm up on a side stream (required before capture), then restore the state
            backup = [t.clone() for t in self._state_tensors()]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            # the graph runs in float32 (TF32 if enabled): launch overhead, not
            # precision, limits speed at batch size 1, and autocast's weight cache
            # doesn't mix with graph capture
            with torch.cuda.stream(stream), torch.autocast(device_type="cuda", enabled=False):
                for _ in range(2):
                    self._step_body()
            torch.cuda.current_stream().wait_stream(stream)
            for t, b in zip(self._state_tensors(), backup):
                t.copy_(b)
            del backup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast(device_type="cuda", enabled=False):
                self.graph_logits = self._step_body()
            self.graph = graph
        except Exception as e:  # e.g. an unsupported operation or driver issue
            print(f"CUDA graph capture failed ({e}); running without it", file=sys.stderr)
            self.cuda_graph = False
            self.graph = None

    @torch.no_grad()
    def step(self, token):
        """Feed one token; returns the logits (vocab,) for the next token."""
        self.token_t.fill_(token)
        if self.cuda_graph and self.graph is None:
            self._capture()
        if self.graph is not None:
            self.graph.replay()
            logits = self.graph_logits.clone()  # the graph's output buffer is reused
        else:
            logits = self._step_body()
        self.pos += 1
        self.ids.append(token)
        self.last_logits = logits
        return logits

    @torch.no_grad()
    def feed(self, ids):
        """Feed tokens after everything so far (no reset); returns the logits after the last one."""
        logits = self.last_logits
        for token in ids:
            logits = self.step(token)
        return logits

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
                self.S[layer].copy_(torch.matmul(q[:, T - keep :].transpose(1, 2), v[T - keep :]))
                if self.capacity:
                    # oldest first in slots 0..keep-1; the next write goes to slot keep % capacity
                    self.ring_q[layer][:keep].copy_(q[:, T - keep :].transpose(0, 1))
                    self.ring_v[layer][:keep].copy_(v[T - keep :])
            y_kv = m.ln(y_kv)
            y_sparse = F.relu(y_kv @ m.encoder_v)
            xy_sparse = x_sparse * y_sparse
            y_mlp = xy_sparse.transpose(1, 2).reshape(1, 1, T, -1) @ m.decoder
            x = m.ln(x + m.ln(y_mlp))
        self.pos = T
        self.pos_t.fill_(T)
        if self.capacity:
            self.slot_t.fill_(keep % self.capacity)
        self.ids = list(ids)
        self.last_logits = x[0, 0, -1] @ m.lm_head
        return self.last_logits


@torch.no_grad()
def generate_stream_recurrent(
    model, ids, max_new_tokens, temperature=1.0, top_k=None, context_size=None, state=None, cuda_graph=False,
    resume=False,
):
    """Like inference.generate_stream, but recurrent. ids is a list of prompt token ids.

    context_size is the sliding window (None: unlimited). Pass a RecurrentBDH as
    state to reuse it, including a recorded CUDA graph. It is reset first, unless
    resume=True: then ids are fed after what the state already holds (ids may be
    empty to continue straight from it, e.g. after load()).

    Each yielded token is fed into the state only when the next one is requested,
    so if the caller stops early, the last yielded token is not in the state.
    """
    rec = state if state is not None else RecurrentBDH(model, window=context_size, cuda_graph=cuda_graph)
    logits = rec.feed(ids) if resume else rec.prefill(ids)
    if logits is None:
        raise ValueError("nothing to continue from: the state is empty and no tokens were given")
    for i in range(max_new_tokens):
        token = sample_next(logits.float().unsqueeze(0), temperature, top_k).item()
        yield token
        if i + 1 < max_new_tokens:
            logits = rec.step(token)
