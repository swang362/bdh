# Benchmarks

Results from training BDH on TinyStories with two tokenizers: raw bytes, and a 4096-token SentencePiece BPE vocabulary.

Each measurement is marked as one of:
- **Measured:** printed by the scripts.
- **Derived:** calculated from measured values.
- **Estimated:** depends on an assumption, which is stated.

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| Dataset | TinyStories V2 (GPT-4), prepared with `prepare_data.py` |
| Model | Default BDH config: 6 layers (shared weights), `n_embd` 256, 4 heads, `mlp_mult` 128 |
| Batch | 32 sequences × 512 tokens = 16,384 tokens per step |

The exact training flags weren't recorded, beyond the step counts below.

## Results

| | Bytes | SentencePiece 4096 |
|---|---|---|
| Checkpoint | `checkpoints/tinystories/latest.pt` | `checkpoints/ts_sp4096/latest.pt` |
| Parameters (measured) | 25,296,896 | 27,262,976 |
| Training steps (measured) | 40,000 | 10,000 |
| Training time (measured) | about 3,600s | about 1,000s |
| Tokens seen (derived) | 655M | 164M |
| Text seen (estimated, at 3.5–4 bytes/token for SentencePiece) | 655M bytes (about 0.29 epoch) | about 575–655M bytes |
| Training throughput (derived) | about 182K tokens/s | about 164K tokens/s |
| Generation speed (measured) | 63.2 tokens/s = 63 chars/s | 54.6 tokens/s ≈ 215 chars/s (derived) |
| First-token latency (measured) | 535ms | 511ms |
| Peak GPU memory, generation (measured) | 834.6MB | 570.7MB |

## Samples

Both use the prompt `One day`, with `top-k` 3 and temperature 1.0 (the `inference.py` defaults).

### Bytes, 40K steps, 500 tokens

```
One day, a big bear named Ben and a little bird named Lily were in the forest. They wanted to discuss what to do. Ben said, "I will help you find your toy." Lily smiled and said, "Okay, let's go!"
As they walked, they saw a big tree. They talked and laughed as they looked for the toy. Suddenly, the tree started to shake! The tree fell down, and Ben and Lily fell to the ground. They were scared, but Ben was nice.
"Are you okay?" asked Lily. "Yes, I am," said Ben. They both smiled and said goodbye. From
```

- **Stopped at:** the 500-token limit, partway through a sentence.
- **Strong:** correct spelling and punctuation, well-formed dialogue, names stay consistent.
- **Weak:** it mentions a toy that was never introduced, "They were scared, but Ben was nice" doesn't follow logically, and the ending comes abruptly.

### SentencePiece 4096, 10K steps, up to 200 tokens

```
One day, a little girl named Lily went to the store with her mom. She saw a pretty dress and wanted to wear it. Lily asked her mom if she could have the dress. Her mom said yes, and Lily was very happy.
Lily wore the dress to the park with her mom. She played on the swings, the slide, and the seesaw. She had lots of fun at the park. When it was time to go home, Lily was tired but very happy.
```

- **Stopped at:** `<|endoftext|>` after 99 tokens.
- **Strong:** the events follow logically from start to finish, and the story has a complete arc and a natural ending.
- **Weak:** the plot is simple and has no conflict.

## Findings

1. **SentencePiece reached similar or better quality with about 3.6× less training time.** Training throughput in tokens was nearly the same: the 4096-token output layer costs about 10%. But each token carries about 3.5–4 bytes of text, so both runs saw roughly the same amount of text, and SentencePiece did it in about a quarter of the time.
2. **More text per training window improves coherence.** 512 tokens cover about 1,800–2,000 characters with SentencePiece, versus 512 with bytes, so whole stories fit in one window. The SentencePiece sample holds together better despite less compute.
3. **Text generation is about 3.4× faster** in characters per second, for the same reason.
4. **The end-of-text marker works end to end.** The SentencePiece model learned where stories end, and `inference.py` stopped there.

## Compute compared with GPT

The README reports that BDH matches GPT-2 at *equal parameter counts*. Parameter count understates BDH's compute, though: the weights are shared across layers, and the sparse dimension is wide (N = 8,192 per head).

| Forward pass, per token (estimated, T = 512) | BDH (this config) | GPT with 25M parameters |
|---|---|---|
| Projections | about 300M FLOPs | about 50M FLOPs |
| Attention | about 200M FLOPs | about 10–20M FLOPs |
| **Total** | **about 500M FLOPs** | **about 60–70M FLOPs** |

**Consistency check:** 3 × 500M FLOPs per token for training × 182K tokens/s ≈ 275 TFLOP/s, about 28% of the H100's bf16 peak. That's a typical utilization, so the estimate holds together.

**What this means:** at equal parameters, BDH uses about 8× the compute of a GPT. At equal wall-clock time, a GPT could process several times more tokens, or be much larger. The runs above don't include a GPT baseline, so **they don't show whether BDH beats GPT under either comparison.**

## Limitations

- **One sample per model.** Quality differences between single samples are anecdotal.
- **No validation loss.** `train.py` doesn't evaluate on validation data yet. Training loss, which is logged as bits per byte, is the only numeric comparison available.
- **Unequal budgets.** The runs differ in both steps and wall-clock time; the comparison is roughly at equal text seen.
- **No GPT baseline.** The GPT compute figures are estimates, not measurements.
- **Slow generation for both models.** It's limited by launching GPU operations (one CPU core at 100%), and it re-processes the full 512-token context for every new token. The attention has no softmax, so it could run as a recurrent state instead, which would make each token cost the same regardless of context length. That isn't implemented yet.

## Next steps

1. **SentencePiece for 40K steps (about 1 hour):** equal wall-clock time with the byte run, about 2.3B bytes of text, roughly one epoch.
2. **Validation evaluation:** report validation bits per byte for each checkpoint.
3. **GPT baseline:** same data, tokenizer and schedule. Compare at equal parameters (about 25M) and at equal wall-clock time.
4. **Activation sparsity:** measure the share of non-zero values in `x_sparse` and `y_sparse`. That tests BDH's claim of interpretable activations.
