# Plan: longer context for BDH

**Goal:** make BDH models use context well beyond the 512-token training block, and give `chat.py` a memory of long conversations.

Status: **plan only**; nothing here is implemented yet. Estimates are rough, and the phases include measuring them.

## Where things stand

| | Today |
|---|---|
| Training context | 512 tokens (`--block-size`), about 1,800–2,000 characters with SentencePiece |
| Default inference (`--context-size` = block size) | Sliding window: the model sees the last 511 tokens and forgets anything older ([recurrent.md](recurrent.md)) |
| Unlimited mode (`--context-size 0`) | Every token is added to the state, but quality degrades past 512 tokens |
| Chat | The conversation stays in the recurrent state. Older turns beyond the window no longer influence replies, although the full history is saved |

### Why unlimited mode doesn't simply work

1. **The model never learned to use distant tokens.**
   - **Position encoding:** RoPE encodes distance as rotation angles, and the angles for distances past 511 were never seen in training.
   - **The mix:** each read of the state blends far more tokens than in training. The LayerNorm after attention rescales the size of that sum, but not its content.
2. **A fixed-size state compresses.** S has a fixed size: 8,192 × 256 per head and layer, about 50M numbers for the default model. Every token is added to the same memory, so with many tokens their contributions interfere and details blur. BDH's large, sparse, positive keys reduce the interference, but can't remove it.
3. **Pure addition never forgets.** Old, irrelevant text keeps full weight forever. Long-context designs usually add **decay** or **gating**; BDH's attention has neither.

The practical rule: **a model uses the context lengths it was trained on.** Training at 2,048 tokens teaches about 2,048, not 10,000.

## Targets

| Target | Approach | Realistic? |
|---|---|---|
| **About 2K tokens** of usable context | Fine-tune at `--block-size 2048` | Yes: a direct, testable improvement |
| **4–8K tokens** | Chunked attention for training, then fine-tune at 4–8K | Yes, with the chunked-attention work |
| **10K+ tokens** of detailed recall in the state | Architecture changes (decay, gating, delta rule) and long training data | Research territory |
| **Chat memory of any length** (100K+ tokens) | **Retrieval** over the saved transcript, with recent turns in the window | Yes, and it works with current checkpoints |

## Phases

### Phase 1: measure (length evaluation)

A script, `length_eval.py`, measuring how quality changes with distance:
- **Loss by position:** average loss at positions 0–512, 512–1K, 1–2K, 2–4K and 4–8K on long held-out documents (Wikipedia articles, concatenated if needed). Measured in **window mode** and **unlimited mode**, using `recurrent.py` so any length fits in memory.
- **Needle in a haystack:** a short fact ("The secret code is 4172.") placed at depths of 10%, 50% and 90% into 1K–8K tokens of filler text, followed by a question. Scored as exact-match accuracy, for pretrained and chat-tuned checkpoints.
- **Output:** a table and a plot per checkpoint, in `docs/`.

This is the baseline for everything after. It answers whether unlimited mode helps or hurts at all today, and by how much.

*Effort:* about 1 day. *Compute:* minutes per checkpoint.

### Phase 2: RoPE position scaling at inference (cheap experiment)

- An option such as `--rope-scale 0.25` multiplies positions by the factor, so 2,048 tokens look like 512 to the position encoding. This is the "position interpolation" trick used to extend transformers.
- **Test with phase 1's evaluation.** It addresses the position part of problem 1, not the mixing or capacity problems.
- **Expectation:** a partial improvement at most, without fine-tuning. Mainly useful to separate *position* effects from *capacity* effects.

*Effort:* half a day, a small change in `recurrent.py` and `bdh.py`'s phase computation, plus evaluation.

### Phase 3: fine-tune at 2,048 tokens

- **Continue training** the Wikipedia checkpoint with `train.py --block-size 2048`. Resuming with a new block size is already supported. Use `--grad-accum` to fit memory, and keep tokens per step at about 16K, e.g. `--batch-size 8`.
- **Data: Wikipedia or FineWeb-Edu,** whose documents are long. **Not TinyStories:** its stories are about 200 tokens, so long blocks would be unrelated stories, and there'd be nothing far away to learn from.
- **Budget:** try 2K–5K steps first, then more if the loss past 512 keeps improving.
- **Cost:** at 2,048 tokens, attention costs about 4× more per token than at 512, so a step is roughly 2× slower overall (an estimate).
- **Chat:** fine-tune the new checkpoint again with `finetune.py --block-size 2048`, ideally with some longer multi-turn conversations, e.g. by concatenating Dolly and Alpaca examples into multi-turn sessions.
- **Evaluation:** phase 1 again. Success means lower loss at positions 512–2K than the original checkpoint, needle accuracy at 2K clearly above the original, and no regression below 512.

*Effort:* 1–2 days. *Compute:* several hours on the H100.

### Phase 4: chunked causal attention for training

**Implemented:** `train.py --attn-chunk N` and `finetune.py --attn-chunk N`, checked by `test_chunked.py`. See [train.md](train.md#chunked-attention-for-long-blocks---attn-chunk). Two corrections to the plan below: it's **slower** than full attention at 512 tokens, because reading and writing the state costs about `2 × n_embd` tokens' worth of attention per token. It pays off above roughly `chunk + 2 × n_embd` tokens, about 1,300 at `--n-embd 512`. And the backward pass keeps only **one** state per layer, rebuilding earlier states by subtraction, so the memory question below was solved without activation checkpointing.

The parallel attention computes a full T × T score matrix, so its cost grows with the **square** of the length. The chunked form processes the block in chunks (e.g. 256 tokens): attention inside each chunk is parallel, and a running state S carries earlier chunks forward, the same state as in [recurrent.py](../recurrent.py). Its cost grows **linearly** with length.
- **The same model mathematically:** a drop-in replacement for training, checked against the parallel form with tests like [test_recurrent.py](../test_recurrent.py).
- **About 15–20% faster even at 512 tokens,** since it skips the upper half of the score matrix. Much faster at 4K+ tokens.
- **Memory:** the state per chunk boundary has to be kept for backpropagation, which is the main design question. Options: keep the states, or recompute them during backprop (activation checkpointing).

*Effort:* 3–5 days, including tests.

### Phase 5: train at 4–8K tokens

- With phase 4, continue from the phase 3 checkpoint at `--block-size 4096`, then 8192.
- **Data:** long documents only, e.g. Wikipedia articles above a length threshold, or FineWeb-Edu long documents.
- **Evaluation:** phase 1, extended to 16K, to see how far past the training length quality holds.

*Effort:* 1–2 days of work. *Compute:* a day or more of H100 time.

### Phase 6: retrieval memory for chat (in parallel with phases 3–5)

It works with **current** checkpoints, for conversations of any length:
1. `chat.py` already keeps the full message history, and saves it in the state file.
2. For each new question, **retrieve the most relevant earlier turns** that have left the window: by keyword overlap (BM25) first, optionally embeddings later.
3. Build the prompt from **the retrieved turns plus the recent turns**, within the window, e.g. as a `### Memory:` section before the conversation. Fine-tuning with that format teaches the model to use it.
4. **Evaluation:** a long-conversation test. Facts stated early (a name, a number) are asked about thousands of tokens later, and exact-match accuracy is scored against a window-only baseline.

This gives exact recall of **relevant** facts from any length. The state still handles the recent flow of the conversation.

*Effort:* 2–3 days. *Compute:* negligible.

### Phase 7 (research, optional): architecture changes for 10K+ in the state

Only if phases 1–5 show that capacity or forgetting, rather than training length, is the limit. Options:
- **Decay:** S ← γ·S + kᵀv, with a fixed γ slightly below 1, as in RetNet. Old content fades, and the scale stays bounded.
- **Gating:** a learned, input-dependent γ, as in Mamba and Gated DeltaNet. The model decides what to keep.
- **Delta rule:** S ← S + kᵀ(v − k·S). Each write partly replaces what was stored under a similar key, instead of piling up, which improves recall capacity (DeltaNet).

All of these **change the model:** they need training from scratch or substantial retraining, aren't compatible with current checkpoints, and move away from the paper's design. Each one would be a separate experiment, compared with phase 5 on phase 1's evaluation.

## Success criteria

| Phase | Criterion |
|---|---|
| 1 | Baseline table and plots for current checkpoints, window vs. unlimited |
| 2 | A clear answer to whether position scaling alone helps |
| 3 | Loss at 512–2K below the original checkpoint's; needle accuracy at 2K well above; no regression below 512 |
| 4 | Chunked and parallel training match within rounding; faster at 512, and much faster at 4K+ |
| 5 | Usable context up to about the training length (4–8K), measured by phase 1 |
| 6 | Long-conversation fact recall clearly above window-only, at conversation lengths far beyond the window |

## Risks

- **Short documents:** long-context training only helps if the data has long documents where early text matters. Check the length distribution before phase 3.
- **Forgetting short-context skills:** fine-tuning on long blocks can hurt quality at short lengths. Watch the loss at 0–512 in phase 1's evaluation.
- **Memory at long lengths:** activations grow with block length. Use `--grad-accum`, and activation checkpointing if needed.
- **Capacity:** the state has a fixed size; phases 3–5 may show a ceiling. Phase 6 is the practical fallback, and phase 7 the research one.
- **Chat data:** long multi-turn conversations are scarce; they may need to be constructed from shorter ones.

## Recommended order

1. **Phase 1:** measure.
2. **Phase 3:** fine-tune at 2K and measure again. Phase 2 can be tried quickly along the way.
3. **Phase 6** for chat, since it's useful regardless of how the others turn out.
4. **Phases 4 and 5** if phase 3 works well.
5. **Phase 7** only if the results point to state capacity as the limit.
