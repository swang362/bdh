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
| `--batch-size N` | 32 | Sequences per step |
| `--block-size N` | 512 | Sequence length in tokens. Saved in the checkpoint and used by `inference.py` as its context window |
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
- **ms/step and tok/s:** throughput. The first window includes `torch.compile` warmup, and checkpoint saving isn't counted.

## Checkpoints, resuming and snapshots

- **What a checkpoint contains:** the weights, optimizer and gradient-scaler state, the step, the model config, the tokenizer, `block_size` and the random-number state. Every save writes a temp file first, so an interrupted save can't corrupt `latest.pt`.
- **Ctrl+C** saves `latest.pt` before exiting. Run the same command again to continue.
- **What resuming requires:** the same model options (`--n-layer`, `--n-embd`, `--n-head`, `--dropout`, `--mlp-mult`) and the same tokenizer. Otherwise it stops with an error; use a new `--ckpt-dir` or `--no-resume`.
- **Options you can change when resuming:** `--lr`, `--min-lr`, `--weight-decay`, `--batch-size`, `--block-size` and `--max-iters`. The learning rate is recomputed from the step number. Changing `--max-iters` reshapes the rest of the schedule, and the rate can jump back up.
- **Snapshots** are full checkpoints. Resume from one by copying it over `latest.pt`, or use it directly with `inference.py --checkpoint`. Each snapshot is about 3× the size of the weights, because it includes AdamW's state: about 290MB for the default model. Nothing deletes old snapshots automatically.

## Recipes

```
# Tiny Shakespeare sanity check (small data: keep training short and use more dropout)
python train.py --max-iters 3000 --dropout 0.2 --ckpt-dir checkpoints/shakespeare

# TinyStories, about 650M tokens
python train.py --data-dir data/tinystories_sp4096 --ckpt-dir checkpoints/ts_sp4096 \
    --max-iters 40000 --warmup-iters 1000 --dropout 0.0 --ckpt-freq 2000 --log-freq 200

# Out of GPU memory: halve the batch size (tokens per step halve too)
python train.py ... --batch-size 16

# Longer context with the same tokens per step
python train.py ... --block-size 1024 --batch-size 16

# Continue a finished run at a lower learning rate
python train.py ... --max-iters 50000 --lr 3e-4

# Old behavior: constant learning rate, no warmup
python train.py --lr-schedule constant --warmup-iters 0
```

## Tips

- **Memory:** the main activation per layer has shape `batch × heads × block × N`. At the defaults that's 32 × 4 × 512 × 8192 = 537M values, about 1GB in bf16 for each such tensor. Lower `--batch-size` first.
- **How long to train:** a common rule of thumb is about 20 tokens per parameter, roughly 500M tokens for 25M parameters. Tokens per step are `batch-size × block-size`, 16,384 by default.
- **Plateaus:** if the loss stops falling, check the learning rate first. A cosine decay to `--min-lr` usually gives a further drop near the end. On small datasets like Tiny Shakespeare, a low training loss often means memorization.
