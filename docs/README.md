# Command guide

This folder documents the command-line scripts for training and running BDH models.

| Script | Purpose | Guide |
|---|---|---|
| `prepare_data.py` | Download or split a dataset, optionally train a tokenizer, write `train.bin` / `val.bin` | [prepare_data.md](prepare_data.md) |
| `train.py` | Train a model, with checkpointing, resuming, snapshots and an LR schedule | [train.md](train.md) |
| `inference.py` | Generate text from a checkpoint | [inference.md](inference.md) |
| `probe.py` | Score fact recall of checkpoints with greedy completions | [probe.md](probe.md) |
| `finetune.py` | Fine-tune a checkpoint on question-and-answer data for chat | [chat.md](chat.md) |
| `chat.py` | Chat with, or ask questions to, a fine-tuned checkpoint | [chat.md](chat.md) |

Each script also prints its full option list with `--help`.

For measured results, see [benchmarks.md](benchmarks.md).

## Setup

```
pip install -r requirements.txt
```

`sentencepiece` is only needed for the SentencePiece tokenizer, and `pyarrow` only for the `wikipedia` and `fineweb-edu` datasets. `psutil` is optional: it enables the RAM line in `inference.py`'s resource report on Windows.

## Quick start

**Smallest run.** Tiny Shakespeare, byte-level, nothing to prepare:

```
python train.py
python inference.py "To be or "
```

`train.py` downloads `input.txt` on the first run.

**Recommended for text completion.** TinyStories with a 4096-token BPE tokenizer:

```
python prepare_data.py tinystories --tokenizer sentencepiece --vocab-size 4096
python train.py --data-dir data/tinystories_sp4096 --ckpt-dir checkpoints/ts_sp4096 \
    --max-iters 40000 --warmup-iters 1000 --dropout 0.0 \
    --prompt "Once upon a time" --sample-tokens 300
python inference.py "Once upon a time" --checkpoint checkpoints/ts_sp4096/latest.pt
```

**Quick experiment on a subset:**

```
python prepare_data.py tinystories --max-train-bytes 100_000_000 --name ts_100mb
python train.py --data-dir data/ts_100mb --ckpt-dir checkpoints/ts_100mb --max-iters 5000
```

**Knowledge-heavy text.** Wikipedia, 2 shards, 16K BPE vocabulary:

```
python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece --vocab-size 16384
python train.py --data-dir data/wikipedia_2shards_sp16384 --ckpt-dir checkpoints/wiki_sp16384 \
    --max-iters 40000 --warmup-iters 1000 --dropout 0.0 \
    --prompt "The history of" --sample-tokens 300
```

Expect fluent but factually unreliable text, and a higher bits per byte than on TinyStories. See the tips in [prepare_data.md](prepare_data.md).

## Directory layout

```
data/
  raw/<dataset>/            cached downloads, shared by all tokenizers
  <name>/                   one prepared dataset
    train.bin  val.bin      token ids (uint8 for bytes, uint16 for sentencepiece)
    meta.json               tokenizer, vocab size, dtype, token counts, bytes/token
    tokenizer.model         sentencepiece datasets only
checkpoints/
  latest.pt                 most recent checkpoint, used for resuming
  best.pt                   lowest validation loss so far
  step0005000_bpb0.5123.pt  snapshots kept every --snapshot-freq steps (named by training bits per byte)
```

`data/`, `checkpoints/` and `input.txt` are git-ignored.

## Rules that apply across scripts

- **Use one `--ckpt-dir` per experiment.** `train.py` resumes from `<ckpt-dir>/latest.pt` automatically. It refuses to resume if the model options or the tokenizer differ from the checkpoint.
- **Checkpoints are self-contained.** They store the model config, the tokenizer and the training block size, so `inference.py` needs only `--checkpoint`.
- **Compare runs in bits per byte, not loss.** Loss per token depends on the tokenizer. `train.py` logs both, e.g. `loss 2.1 (0.78 bpb)`.
