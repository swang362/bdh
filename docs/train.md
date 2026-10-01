# train.py

`train.py` trains a BDH model. It saves checkpoints as it goes and resumes automatically. The learning rate warms up and then follows a cosine decay, and a short text sample is generated at the end.

```
python train.py [options]
```

## Data sources

| How | Tokenizer | Validation split |
|---|---|---|
| Default: `input.txt` (Tiny Shakespeare, downloaded if missing) | bytes | The last 10% of the file |
| `--data FILE`: any text file | bytes | The last 10% of the file |
| `--data-dir DIR`: a folder made by `prepare_data.py` | from `meta.json` | `val.bin` |

With `--data-dir`, the tokenizer, vocab size and data type all come from the dataset's `meta.json`, so there's nothing else to set.

## Options

### Training

| Option | Default | Description |
|---|---|---|
| `--max-iters N` | 3000 | Total training steps. Also where the cosine schedule ends |
| `--batch-size N` | 32 | Sequences per optimizer step |
| `--grad-accum N` | 1 | Split each step into N micro-batches of `batch-size / N` sequences. Tokens per step stay the same, activation memory drops about N×, and each step takes somewhat longer. `--batch-size` must be divisible by N |
| `--block-size N` | 512 | Sequence length in tokens. Saved in the checkpoint and used by `inference.py` as its context window |
| `--attn-chunk N` | 0 (full attention) | Compute attention in chunks of N tokens with a running state, for long blocks. Same model and results; see [Chunked attention](#chunked-attention-for-long-blocks---attn-chunk) |
| `--lr LR` | 1e-3 | Peak learning rate, reached at the end of warmup |
| `--lr-schedule` | `cosine` | `cosine` decays to `--min-lr` by `--max-iters`; `constant` stays at `--lr` after warmup |
| `--warmup-iters N` | 200 | Steps to ramp up linearly from about 0 to `--lr` |
| `--min-lr LR` | `lr / 10` | Final learning rate of the cosine schedule |
| `--weight-decay W` | 0.1 | AdamW weight decay |
| `--seed N` | 1337 | Random seed. Ignored when resuming, because the saved random state is restored |
| `--compile` / `--no-compile` | on | Use `torch.compile`. Turn it off if it fails, which is common on Windows without Triton |
| `--data FILE` | `input.txt` | Plain text training file, read as bytes; ignored when `--data-dir` is set |
| `--data-dir DIR` | none | Dataset folder made by `prepare_data.py` |

### Model

| Option | Default | Description |
|---|---|---|
| `--n-layer N` | 6 | Number of layers. **All layers share the same weights**, so this changes compute, not parameter count |
| `--n-embd D` | 256 | Embedding dimension |
| `--n-head H` | 4 | Number of heads |
| `--dropout P` | 0.1 | Dropout rate. Use 0.0 on large datasets trained for under one epoch |
| `--mlp-mult M` | 128 | Sparse dimension `N = M × D / H` |

The parameter count is about `3 × 128 × D² + 2 × vocab × D` at the default `--mlp-mult`. That's 25.3M for the default byte model, and it's printed at startup.

### Logging and checkpoints

| Option | Default | Description |
|---|---|---|
| `--log-freq N` | 100 | Print a log line every N steps |
| `--ckpt-freq N` | 500 | Save `<ckpt-dir>/latest.pt` every N steps |
| `--ckpt-dir DIR` | `checkpoints` | Checkpoint folder. Use a separate one per experiment |
| `--snapshot-freq N` | 5000 | Every N steps, and at the end, also keep a copy named `step<N>_bpb<B>.pt`, where B is the recent training bits per byte. `0` turns this off |
| `--resume` / `--no-resume` | on | Resume from `<ckpt-dir>/latest.pt` if it exists |

### Validation

| Option | Default | Description |
|---|---|---|
| `--eval-freq N` | 500 | Evaluate on the validation split every N steps and at the end. Saves `<ckpt-dir>/best.pt` whenever validation loss improves. `0` turns this off |
| `--eval-iters N` | 50 | Validation batches averaged per evaluation. With the defaults that's 50 × 32 × 512 = about 820K tokens |

### Sample after training

| Option | Default | Description |
|---|---|---|
| `--prompt TEXT` | `To be or ` | Prompt for the sample |
| `--sample-tokens N` | 100 | Tokens to generate; `0` skips the sample |

## Reading the log

```
Step: 2000/40000 loss 1.62 (2.34 bpb) | lr 9.98e-04 | 142.3 ms/step | 115,130 tok/s | elapsed 290.1s
```

- **loss:** training cross-entropy per token, averaged since the previous log line. Dropout is on during training, so it reads slightly high.
- **bpb:** the same loss in bits per byte of text. Use it to compare runs with different tokenizers.
- **lr:** the learning rate at this step.
- **ms/step and tok/s:** throughput. The first window includes `torch.compile` warmup. Checkpoint saving and evaluation aren't counted.

Every `--eval-freq` steps, a validation line follows:

```
Eval step 2000: val loss 1.6543 (2.3867 bpb) | new best
Eval step 2500: val loss 1.6601 (2.3951 bpb) | best 1.6543 (2.3867 bpb)
```

- **val loss and bpb:** measured on the validation split with dropout off, so they're the numbers to compare runs by.
- **The same batches each time:** every evaluation uses the same validation batches (a fixed seed), so changes between evaluations reflect the model, not which batches were drawn.
- **Overfitting:** training loss still falling while validation loss rises means the model is memorizing. That's common on Tiny Shakespeare.

## Checkpoints, resuming and snapshots

- **What a checkpoint contains:** the weights, optimizer and gradient-scaler state, the step, the model config, the tokenizer, `block_size`, the best validation loss so far and the random-number state. Every save writes a temp file first, so an interrupted save can't corrupt `latest.pt`.
- **`best.pt`** is the checkpoint with the lowest validation loss so far, saved at evaluation time. It's usually the one to use with `inference.py --checkpoint`, especially if later training overfits. A resumed run remembers the best validation loss, so `best.pt` is only replaced by something better.
- **Ctrl+C** saves `latest.pt` before exiting. Run the same command again to continue.
- **What resuming requires:** the same model options (`--n-layer`, `--n-embd`, `--n-head`, `--dropout`, `--mlp-mult`) and the same tokenizer. Otherwise it stops with an error; use a new `--ckpt-dir` or `--no-resume`.
- **Options you can change when resuming:** `--lr`, `--min-lr`, `--weight-decay`, `--batch-size`, `--block-size`, `--attn-chunk` and `--max-iters`. The learning rate is recomputed from the step number. Changing `--max-iters` reshapes the rest of the schedule, and the rate can jump back up.
- **Snapshots** are full checkpoints. Resume from one by copying it over `latest.pt`, or use it directly with `inference.py --checkpoint`. Each snapshot is about 3× the size of the weights, because it includes AdamW's state: about 290MB for the default model. Nothing deletes old snapshots automatically.

## Recipes

```
# Tiny Shakespeare sanity check (small data: keep training short and use more dropout)
python train.py --max-iters 3000 --dropout 0.2 --ckpt-dir checkpoints/shakespeare

# TinyStories, about 650M tokens
python train.py --data-dir data/tinystories_sp4096 --ckpt-dir checkpoints/ts_sp4096 \
    --max-iters 40000 --warmup-iters 1000 --dropout 0.0 --ckpt-freq 2000 --log-freq 200

# Out of GPU memory: split each step into micro-batches (tokens per step stay the same)
python train.py ... --grad-accum 4

# Larger model that needs gradient accumulation (32 sequences per step as 8 micro-batches of 4)
python train.py ... --n-embd 1024 --lr 3e-4 --grad-accum 8

# Longer context with the same tokens per step
python train.py ... --block-size 1024 --batch-size 16

# Long blocks: chunked attention (see below)
python train.py ... --block-size 4096 --batch-size 8 --attn-chunk 256

# Continue a finished run at a lower learning rate
python train.py ... --max-iters 50000 --lr 3e-4

# Old behavior: constant learning rate, no warmup
python train.py --lr-schedule constant --warmup-iters 0
```

## Chunked attention for long blocks (`--attn-chunk`)

Full attention computes a `block × block` score matrix per head and layer, so its cost grows with the **square** of the block size. With `--attn-chunk N`, attention is computed N tokens at a time:
- **Inside a chunk,** directly, as before.
- **Earlier chunks** contribute through a running state S = Σ kᵀv, the same state as recurrent generation ([recurrent.md](recurrent.md)). Each chunk reads S, then adds its own tokens to it.

It's **the same model**: it computes the same values in a different order, so checkpoints, resuming and inference are unaffected. You can turn it on or off when resuming. `test_chunked.py` checks it against full attention, forward and backward.

**When it pays off.** Per token and head, full attention costs about `block × N` multiply-adds (N is the sparse dimension). The chunked form costs about `(chunk + 2 × n_embd) × N`: the chunk itself, plus reading and writing the state, which is `N × n_embd` per head. So it's faster when the block size is above roughly `chunk + 2 × n_embd`, and the gain grows with the block size:

| `--n-embd`, `--attn-chunk` | Break-even block size (estimate) | Attention cost at 2048 | at 4096 | at 8192 |
|---|---|---|---|---|
| 256, 128 | about 640 | about 0.3× | 0.16× | 0.08× |
| 512, 256 | about 1,300 | about 0.6× | 0.3× | 0.16× |

At 512 tokens it's **slower** than full attention for these widths, so leave it off there. Measure on your GPU with `python test_chunked.py --bench --n-embd 512`, which times a training step and peak memory with full and chunked attention for several block sizes and chunk sizes.

**Memory.** No `block × block` matrices are stored, and the backward pass keeps only one state per layer (micro-batch × heads × N × `n_embd` float32 values, about 0.5GB for 4 sequences at `--n-embd 512`): it rebuilds earlier states by subtracting each chunk's contribution. The largest activations, of shape `micro-batch × heads × block × N`, are unchanged, so `--grad-accum` is still the main memory control.

**Precision.** Under bf16 autocast, the matmuls run in bf16 as before, and the state is summed in float32. Results differ from full attention only by rounding.

**`torch.compile`** skips the chunked function and compiles the rest of the model around it.

## Tips

- **Memory:** the main activation per layer has shape `micro-batch × heads × block × N`. At the defaults that's 32 × 4 × 512 × 8192 = 537M values, about 1GB in bf16 for each such tensor. It grows in proportion to `--n-embd` and `--mlp-mult`: 2× at `--n-embd 512`, 4× at 1024. Use `--grad-accum` first, since lowering `--batch-size` would also change the tokens per step and the training behavior.
- **Choosing `--grad-accum`:** use the smallest N that fits, because each extra micro-batch adds some overhead. Watch the GPU memory in the first few steps (e.g. with `nvidia-smi`), and double N if you run out of memory.
- **How long to train:** a common rule of thumb is about 20 tokens per parameter, roughly 500M tokens for 25M parameters. Tokens per step are `batch-size × block-size`, 16,384 by default.
- **Plateaus:** if the loss stops falling, check the learning rate first. A cosine decay to `--min-lr` usually gives a further drop near the end. Watch the validation loss: if it rises while training loss falls, stop, or use `best.pt`.
- **Fused AdamW:** on CUDA, the optimizer uses PyTorch's fused AdamW, which updates all parameters in a few kernels instead of several per tensor. It's the same algorithm, and checkpoints from before load normally.
- **Evaluation cost:** 50 validation batches every 500 steps adds roughly 3% to training time. The first evaluation also triggers a one-off `torch.compile` for evaluation mode.
