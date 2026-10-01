# Training on FineWeb-Edu

A recipe for training a BDH model on FineWeb-Edu with a **2048-token context window**: a D=512 model of about 134M parameters, trained on about 2.3B tokens in two stages.

Status: **a plan.** Times and memory figures are estimates scaled from the measured D=256 runs in [benchmarks.md](benchmarks.md), not measurements. Check them on your hardware before committing to a long run (see "Before the long run").

## Why FineWeb-Edu

- **Educational web pages** (`HuggingFaceFW/fineweb-edu`, 10B-token sample): explanations, textbooks, course material. More reasoning-style text per token than Wikipedia.
- **Plenty of data:** 14 parquet shards, about 700M tokens each. The model sees each token about once, so dropout isn't needed.
- **Long documents:** many are longer than 512 tokens, so there's real distant context for a longer window to learn from. TinyStories doesn't have this.

## The method

1. **Stage 1: train at 512 tokens** for about 85% of the steps. Short blocks are cheap, and the model learns the language there.
2. **Stage 2: continue at 2048 tokens** for the last 15%. The model learns to use the longer context.

Most long-context models are trained this way. Training at 2048 from the start works too, but costs much more for little gain:

| Block size | Training cost per token (512 = 1×) | 2.3B tokens, all at this length (H100) |
|---|---|---|
| 512 | 1× | about 12–16 h |
| 1024 | about 1.25× | about 15–20 h |
| 2048 | about 1.75× | about 21–28 h |
| 2048 with `--attn-chunk 256` | about 1.4× | about 17–22 h |
| **60K steps at 512, then 10K at 2048 with `--attn-chunk 256`** | | **about 14–17 h** |

Attention cost grows with block length; the rest of the model's cost doesn't. At D=512, attention is about a quarter of the cost at 512 tokens, and more than half at 2048. Chunked attention (`--attn-chunk`, see [train.md](train.md#chunked-attention-for-long-blocks---attn-chunk)) computes the same model with a cost that grows linearly with block length. At D=512 it pays off above about 1,300 tokens, so it's used in stage 2 only.

## Model and settings

| Setting | Value | Why |
|---|---|---|
| `--n-embd` | 512 | 4× the default model's size, and still overnight on one H100 |
| `--n-layer`, `--n-head`, `--mlp-mult` | 6, 4, 128 | The defaults and the paper's settings |
| Vocabulary | 32768 (SentencePiece) | Web text is varied, so a larger vocabulary puts more text in each token. Each 512-token block then holds more context. The extra embedding parameters are cheap at D=512, and ids still fit in uint16 |
| Parameters | about 134M | 3 · 128 · 512² + 2 · 32768 · 512 |
| `--dropout` | 0.0 | Each token is seen about once, so there's nothing to overfit |
| `--lr` | 6e-4 | Lower than D=256's 1e-3: larger models are less stable at the same rate |
| `--warmup-iters` | 2000 | |
| `--weight-decay` | 0.1 | |
| Tokens per step | 32K (64 × 512, then 16 × 2048) | The same in both stages |
| `--grad-accum` | 4 | BDH's sparse activations are large (batch × tokens × 16384 values per head, several per layer), so a full batch doesn't fit in 80GB. Each micro-batch is 8K tokens in both stages |
| Steps | 60,000 + 10,000 | About 2.3B tokens, close to 20 tokens per parameter |

## Commands

**Prepare the data** (4 shards, about 2.8B tokens, with a held-out validation split):

```
python prepare_data.py fineweb-edu --shards 4 --tokenizer sentencepiece --vocab-size 32768
```

Check the output folder name it prints; the commands below assume `data/fineweb-edu_4shards_sp32768`.

**Stage 1: 60,000 steps at 512 tokens**

```
python train.py --data-dir data/fineweb-edu_4shards_sp32768 --ckpt-dir checkpoints/fwe_d512 \
    --n-embd 512 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 4 \
    --max-iters 60000 --lr 6e-4 --warmup-iters 2000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 10000
```

**Stage 2: resume, and continue to 70,000 steps at 2048 tokens**

```
python train.py --data-dir data/fineweb-edu_4shards_sp32768 --ckpt-dir checkpoints/fwe_d512 \
    --n-embd 512 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 2048 --batch-size 16 --grad-accum 4 --attn-chunk 256 \
    --max-iters 70000 --lr 6e-4 --warmup-iters 2000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

Stage 2 resumes from `checkpoints/fwe_d512/latest.pt`, at step 60,000. `--attn-chunk 256` changes only how attention is computed, not the model.

## What happens at the switch

- **Changing the block size on resume works.** `train.py` checks the model options and the tokenizer when it resumes, not the block size. Checkpoints from stage 2 record 2048, so `inference.py`, `chat.py` and `finetune.py` use a 2048-token window automatically.
- **The learning rate ticks up slightly.** Stage 1's cosine schedule ends at 6e-5. Stage 2 resumes partway through a 70,000-step schedule, at about 9e-5, and decays back to 6e-5. A small bump like this is harmless, and it helps the model adapt to the longer blocks.
- **Validation bpb drops at the switch.** With more context, every prediction gets easier, so numbers before and after step 60,000 aren't directly comparable. Compare stage 2 checkpoints only with each other. `best.pt` will likely move to a stage-2 checkpoint at the first evaluation, which is expected.
- **Memory stays about the same:** micro-batches are 8K tokens in both stages. With `--attn-chunk`, no `block × block` score matrices are stored at all.

## Before the long run

1. **Measure the speed.** Run stage 1 for about 200 steps and read tok/s from the log. Hours for stage 1 ≈ 1.97B tokens ÷ tok/s ÷ 3600, and stage 2 is about 1.75× slower per token. Stop with Ctrl+C; running the same command again resumes.
2. **Check memory.** If either stage runs out of memory, double `--grad-accum`. Tokens per step stay the same, so the results don't change.
3. **Check that chunked attention pays off on your GPU:** `python test_chunked.py --bench --n-embd 512` runs the correctness checks, then times training steps with full and chunked attention at several block sizes. Use the fastest chunk size at 2048, or drop `--attn-chunk` if full attention is faster.
4. **Do the same quick check for stage 2** before the real switch: resume a copy of an early checkpoint at `--block-size 2048` in a separate `--ckpt-dir` for a few dozen steps.

## During training

- **Validation bpb should still be falling at step 60,000.** If it flattens early, the model is too small for the data, and the larger variant below is the next step.
- **Loss spikes** that don't recover: resume with a lower `--lr`, e.g. 4e-4. Resuming applies the new learning rate.
- **Snapshots** every 10,000 steps (5,000 in stage 2) keep fallback points, named by bits per byte.

## After training

- **Fact recall:** `python probe.py checkpoints/fwe_d512/best.pt`, and compare with the Wikipedia model. FineWeb-Edu usually helps explanatory text more than fact recall.
- **Long-context use:** whether the model actually uses all 2,048 tokens is measured by the length evaluation in phase 1 of [long_context_plan.md](long_context_plan.md).
- **Chat:** `finetune.py --base checkpoints/fwe_d512/best.pt`. It takes the block size from the base checkpoint, so fine-tuning keeps the 2048 window. See [chat.md](chat.md).
- **Recurrent inference:** with the sliding window, the state grows with the window. At D=512 with a 2048-token window, it's about 4GB: about 0.8GB for S, about 3.2GB for the ring buffer. Saved state files are that size too. Speed per token stays constant. See [recurrent.md](recurrent.md).

## Variants

| | Quick | **This recipe** | Larger |
|---|---|---|---|
| `--n-embd` | 256 | **512** | 768 |
| Vocabulary | 16384 | **32768** | 32768 |
| Parameters | about 34M | **about 134M** | about 276M |
| Shards | 1 | **4** | 7–8 |
| Tokens | about 0.65B | **about 2.3B** | about 5B |
| `--batch-size` / `--grad-accum` (512 tokens) | 64 / 2 | **64 / 4** | 64 / 8 |
| Steps | 20,000 | **60,000 + 10,000** | 150,000 (e.g. 130,000 + 20,000) |
| `--lr` | 1e-3 | **6e-4** | 4e-4 |
| Estimated H100 time | about 1.5 h at 512 | **about 14–18 h** | about 3 days |

**Quick variant**, at 512 tokens only. It matches the Wikipedia model's size, so comparing the two shows the effect of the data alone:

```
python prepare_data.py fineweb-edu --tokenizer sentencepiece --vocab-size 16384

python train.py --data-dir data/fineweb-edu_sp16384 --ckpt-dir checkpoints/fwe_d256 \
    --n-embd 256 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 2 \
    --max-iters 20000 --lr 1e-3 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

**Longer windows.** For 4K–8K tokens, continue from stage 2 with `--block-size 4096` (then 8192), `--attn-chunk 256`, and a `--batch-size` that keeps 32K tokens per step. Chunked attention keeps the attention cost growing linearly. That's phase 5 of [long_context_plan.md](long_context_plan.md).

## What to expect

A 134M-parameter model gives fluent, on-topic educational prose and some basic facts. It won't give reliable answers or multi-step reasoning: those need much larger models and more data.
