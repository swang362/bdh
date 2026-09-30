# Recurrent generation mode

Recurrent mode generates text **one token at a time with a fixed-size memory**, instead of re-reading the whole context for every new token. It's a different way of computing the same model: it uses the weights of any existing checkpoint unchanged, and produces the same output wherever the text fits in the training block size. It's faster, especially for long outputs and conversations, and its memory use doesn't grow with context length.

```
python inference.py "Once upon a time" --recurrent --checkpoint checkpoints/ts_sp4096/best.pt
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

**Measured speed isn't available yet.** Expect a large improvement for long outputs; for short ones, less. Generation at batch size 1 is also limited by the overhead of launching many small GPU operations per token, and recurrent mode still has that overhead. Compare the tok/s that `inference.py` prints with and without `--recurrent` on your GPU.

**State memory,** always in float32 because it's a long running sum:

| Model | S (all layers) | Sliding-window buffers | Total |
|---|---|---|---|
| D=256 (default) | about 200MB | about 400MB | **about 0.6GB** |
| D=512 | about 800MB | about 800MB | about 1.6GB |
| D=1024 | about 3.2GB | about 1.6GB | about 4.8GB |

The window buffers hold the last 511 tokens' keys, so they can be subtracted when they leave. In unlimited mode there are no buffers, only S.

## Where it helps

| Script | Typical length | Benefit |
|---|---|---|
| `inference.py` | 200–1,000+ tokens | **High:** constant speed however long the output |
| `chat.py` | multi-turn conversations, often past 512 tokens | **High:** faster replies in long conversations |
| `probe.py` | about 25 tokens (short prompt plus 10) | **Negligible.** Everything fits in the window, so scores are identical. There's hardly any context to re-read, so speed barely changes. Mainly useful as a check: probe scores with and without `--recurrent` should match. |

In `chat.py`, each reply prepares the state from the whole conversation again. One window's worth is processed in one parallel pass, and anything beyond it token by token. Reusing the state across turns is a possible future improvement.

## Checking it: test_recurrent.py

```
python test_recurrent.py                                                # small random model, CPU, seconds
python test_recurrent.py --checkpoint checkpoints/wiki_sp16384/best.pt  # also your checkpoint (float32)
```

It checks:
1. Feeding tokens one at a time gives the same logits as the parallel forward pass.
2. Parallel prefill followed by steps gives the same logits as the forward pass.
3. With a sliding window, results are identical while everything fits in it.
4. Past the window, parallel prefill and step-by-step agree, and the state never holds more than `window − 1` tokens.

Each check prints the largest difference relative to the largest logit. Anything below 1e-3 passes; float32 rounding typically gives about 1e-6. The script exits with an error code if any check fails.

## Limitations and possible next steps

- **Batch size 1:** one sequence at a time, which is what generation uses.
- **Training still uses the parallel form.** Recurrent mode is for inference. Training on long streams with the state is related to the paper's Section 7.2 (training without backpropagation through time), which isn't implemented.
- **Saving and loading the state** isn't implemented yet. It would let you "remember" a long document or conversation in a fixed amount of memory, and continue later without re-reading it.
- **Reusing the state across `chat.py` turns** isn't implemented yet, as described above.
- **Rounding drift:** the sliding window adds and later subtracts each token's term. In float32 that drift is negligible, even over very long streams.
