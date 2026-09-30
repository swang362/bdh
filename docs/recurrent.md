# Recurrent generation mode

Recurrent mode generates text **one token at a time with a fixed-size memory**, instead of re-reading the whole context for every new token. It's a different way of computing the same model: it uses the weights of any existing checkpoint unchanged, and produces the same output wherever the text fits in the training block size. It's faster, especially for long outputs and conversations, and its memory use doesn't grow with context length.

```
python inference.py "Once upon a time" --recurrent --checkpoint checkpoints/ts_sp4096/best.pt
python inference.py "Once upon a time" --recurrent --cuda-graph --checkpoint checkpoints/ts_sp4096/best.pt
python chat.py --recurrent --checkpoint checkpoints/wiki_chat/best.pt
python probe.py checkpoints/wiki_sp16384/best.pt --recurrent
python test_recurrent.py --checkpoint checkpoints/wiki_sp16384/best.pt
```

## How it works

### The default method re-reads everything

In each layer, BDH's attention ([bdh.py](../bdh.py), `Attention.forward`) computes:

```python
scores = (QR @ KR.mT).tril(diagonal=-1)   # every token vs. every earlier token
return scores @ V
```

For position t that's

  **yₜ = Σₛ₍ₜ (qₜ · kₛ) vₛ**

a weighted sum over all earlier tokens s. Here q and k are the rotated (RoPE) sparse activations, and v is the layer input.

For generation, the default method runs this over the whole context window (512 tokens by default) **for every new token**. That's about 21 billion multiply-adds per token for the default model, nearly all of it recomputing what was already computed for the previous token.

### Recurrent mode keeps a running sum

BDH's attention has **no softmax**, unlike a standard transformer. So the sum can be regrouped:

  yₜ = qₜ · **S**,  with  **S = Σₛ₍ₜ kₛᵀ vₛ**

S is a fixed-size matrix per layer: heads × N × `n_embd`. Each new token adds one term to it:

```python
y = q_t @ S          # read: what earlier tokens contribute
S = S + k_tᵀ v_t     # write: add this token, for the tokens after it
```

Everything else in a BDH layer (encoder, ReLU, `encoder_v`, decoder, LayerNorm) works on each token separately. So the whole model becomes a step function: **(token, state) → (next-token logits, new state)**. The implementation is in [recurrent.py](../recurrent.py).

- **The prompt is processed once,** in parallel: one forward pass, the same computation as training. That builds the initial state, and generation then continues step by step.
- **RoPE position encoding still works.** Each key is rotated by its absolute position when it's added to S. Rotations of q and k combine into a dependence on their distance only, so the results match the default method.

### Relation to the BDH paper

This is the paper's own picture of BDH. S is the **synapse matrix**: adding `kₜᵀ vₜ` strengthens the connection between neurons that are active together, which is a **Hebbian learning rule**. When the paper says BDH's working memory "relies on synaptic plasticity with Hebbian learning", it means this state. Recurrent mode runs BDH the way the paper describes it: a fixed network of neurons whose synapses change as it reads.

## Compatibility and exactness

**Every existing checkpoint works:** pretrained, chat fine-tuned, snapshots, `best.pt`, byte and SentencePiece. Recurrent mode is only a different computation of the same model, so nothing needs retraining or converting.

| Situation | Match with the default method |
|---|---|
| Prompt plus output fit in the context window (the training block size) | **Identical**, apart from tiny rounding differences. `test_recurrent.py` checks this. |
| Longer than the window | **Slightly different, and probably slightly better.** See below. |

**Past the window:** both methods only let each position see the last `block_size − 1` tokens, which is the most a position sees in training. They differ in *how* older tokens are represented:
- **Default method:** every new token re-reads the last 512 tokens *from scratch*. The oldest tokens in the window are recomputed with almost no context of their own.
- **Recurrent mode:** each token's memory entry was computed once, when the token arrived, **with its own full context**. The oldest entry is subtracted from S when it leaves the window.

This works like the sliding-window KV cache in some transformers (e.g. Mistral). Each layer still sums at most 511 tokens, so the sums stay the size seen in training, but the entries are better informed. Through the 6 layers, information can reach up to about 6 × 512 tokens back. It's unlikely to hurt, but it is a change past the window, so compare a few long generations both ways if it matters to you.

**Unlimited context (`--context-size 0`):** nothing is ever removed from S. Memory and compute per token stay constant however long the text gets, which is the architecture's "no context limit" property. But the model only learned to read S up to the training length. Past it, S grows larger than anything seen in training, and **quality degrades**, just as with the default method. A fixed-size state doesn't by itself mean good long-context behavior. That needs training on longer sequences.

## Performance

| Default model (D=256, 6 layers) | Default method, 512-token window | Recurrent |
|---|---|---|
| Compute per new token | about 21 billion multiply-adds | about **300 million**, roughly 70× less |
| Cost as the context grows | grows up to the window | **constant** |
| Memory | activations of the whole window | a fixed state (see below) |

**Measured on an H100** (byte-level TinyStories model, D=256): **276.5 tok/s for 2,000 tokens with `--recurrent`, versus 63.2 tok/s for 500 tokens with the default method**, about 4.5× faster. Details are in [benchmarks.md](benchmarks.md).

It's still limited by the overhead of launching many small GPU operations per token at batch size 1: one CPU core runs at 100% while the GPU mostly waits. `--cuda-graph` targets exactly that; see the next section.

**State memory,** always in float32 because it's a long running sum:

| Model | S (all layers) | Sliding-window ring buffer | Total |
|---|---|---|---|
| D=256 (default) | about 200MB | about 400MB | **about 0.6GB** |
| D=512 | about 800MB | about 800MB | about 1.6GB |
| D=1024 | about 3.2GB | about 1.6GB | about 4.8GB |

The ring buffer holds the last 511 tokens' keys and values, so they can be subtracted when they leave. It's allocated once, up front. In unlimited mode there's no buffer, only S.

## CUDA graphs (`--cuda-graph`)

```
python inference.py "Once upon a time" --recurrent --cuda-graph --checkpoint checkpoints/ts_sp4096/best.pt
python chat.py --recurrent --cuda-graph --checkpoint checkpoints/wiki_chat/best.pt
```

**The problem:** each token step runs about a hundred small GPU operations (6 layers × projections, RoPE, the state read and update, LayerNorms). At batch size 1, each one takes the GPU only a few microseconds. But *launching* each from Python costs about as much or more, so the CPU becomes the bottleneck. That's the `CPU: 100% of one core` in the stats.

**What `--cuda-graph` does:** on the first token step, it records the whole step, every GPU operation in order, as one **CUDA graph**. After that, each token is a single **replay** of the graph: one launch instead of about a hundred, with no Python in between. The GPU runs exactly the same operations, only without waiting for the CPU.

To make this possible, the step was changed to be fully "graph-safe":
- **All state lives in fixed tensors** allocated once: S, the sliding-window **ring buffer**, and the position and slot counters. `reset()` zeroes them in place and never reallocates, because a graph replays fixed memory addresses.
- **No Python decisions inside a step.** The oldest window entry is always subtracted; while the ring isn't full yet, that entry is all zeros, which makes it an exact no-op. Position and ring slot are counters on the GPU, updated by the graph itself.

**Details:**
- **The graph runs in float32** (TF32 where enabled), without mixed precision. At batch size 1 the speed is set by launch overhead, not precision, and mixed precision's weight cache doesn't combine with graph recording.
- **Recorded once per state object.** `chat.py` keeps one state for the whole session and `probe.py` one per checkpoint, so recording happens once and every later reply or probe reuses it. Recording takes a moment on the first token, which shows up as a slightly longer "first token" time.
- **It falls back automatically:** if recording fails, e.g. on an unsupported setup, a warning is printed and generation continues without the graph.
- **CUDA only.** On other devices the flag is ignored, with a message.
- **`test_recurrent.py`** compares graph replay against the reference implementation, including reuse after a reset, whenever a CUDA GPU is available.

**Measured speed** (H100, SentencePiece TinyStories model, D=256, 2,000 tokens): **529.9 tok/s with `--cuda-graph`, versus 261.3 tok/s without**. Excluding startup, that's about 1.6ms instead of about 3.6ms per token. Details are in [benchmarks.md](benchmarks.md). What still runs outside the graph for each token (sampling, `.item()`, decoding and printing) is now the main remaining cost; see the limitations below.

### How this differs from `torch.compile`

| | CUDA graph (`--cuda-graph`) | `torch.compile` |
|---|---|---|
| What it changes | **How** the kernels are launched: all at once, as one replay | **Which** kernels run: it generates new, **fused** kernels, e.g. RoPE's handful of element-wise operations become one |
| Removes launch overhead | Yes, fully: one launch per token | Partly, since fewer kernels means fewer launches. Fully only with `mode="reduce-overhead"`, which **also uses CUDA graphs** underneath |
| Reduces GPU work | No: the same kernels run | Yes: fused kernels read and write memory less often |
| Startup cost | Milliseconds (record once) | Tens of seconds or more (compilation) |
| Requirements | CUDA; fixed shapes and memory, which the recurrent step now has | Triton, which is less mature on Windows and ROCm; it recompiles when shapes change |

They **combine**: compile the step to get fewer, fused kernels, then graph it. At batch size 1, launch overhead is the dominant cost, so CUDA graphs alone capture most of the gain. Once it's gone, the GPU's own work becomes the limit, and then fusing with `torch.compile` would be the next step. That isn't implemented.

## Where it helps

| Script | Typical length | Benefit |
|---|---|---|
| `inference.py` | 200–1,000+ tokens | **High:** constant speed however long the output |
| `chat.py` | multi-turn conversations, often past 512 tokens | **High:** faster replies in long conversations |
| `probe.py` | about 25 tokens (short prompt plus 10) | **Negligible.** Everything fits in the window, so scores are identical. There's hardly any context to re-read, so speed barely changes. Mainly useful as a check: probe scores with and without `--recurrent` should match. |

In `chat.py`, each reply prepares the state from the whole conversation again. One window's worth is processed in one parallel pass, and anything beyond it token by token. The state object itself, with its buffers and any recorded CUDA graph, is kept for the whole session. Carrying the *contents* of the state from one turn to the next is a possible future improvement.

## Checking it: test_recurrent.py

```
python test_recurrent.py                                                # small random model, CPU, seconds
python test_recurrent.py --checkpoint checkpoints/wiki_sp16384/best.pt  # also your checkpoint (float32)
```

It checks:
1. Feeding tokens one at a time gives the same logits as the parallel forward pass.
2. Parallel prefill followed by steps gives the same logits as the forward pass.
3. With a sliding window, results are identical while everything fits in it.
4. Past the window, step-by-step and prefill-plus-steps both match a slow, independent reference. The reference keeps every token in a plain list and explicitly sums the last `window − 1` of them, which checks the ring buffer.
5. With a CUDA GPU: CUDA graph replay matches the reference, also after a reset reuses the same recorded graph. With a GPU, the small random model is tested on both the CPU and the GPU.

Each check prints the largest difference relative to the largest logit. Anything below 1e-3 passes; float32 rounding typically gives about 1e-6. The script exits with an error code if any check fails.

## Limitations and possible next steps

- **Batch size 1:** one sequence at a time, which is what generation uses.
- **Training still uses the parallel form.** Recurrent mode is for inference. Training on long streams with the state is related to the paper's Section 7.2 (training without backpropagation through time), which isn't implemented.
- **Saving and loading the state** isn't implemented yet. It would let you save a session and resume it exactly. With the sliding window, the state only holds the last `window − 1` tokens, so it's a session checkpoint, not a memory of a whole long document.
- **Carrying state contents across `chat.py` turns** isn't implemented yet, as described above.
- **Sampling runs outside the CUDA graph,** as does the `.item()` that sends each token back to Python for printing. Both add a small cost per token.
- **`torch.compile` of the step** (fused kernels) isn't implemented; see the comparison above.
- **Rounding drift:** the sliding window adds and later subtracts each token's term. In float32 that drift is negligible, even over very long streams.
