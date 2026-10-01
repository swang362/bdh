# Training on FineWeb-Edu

Recipes for training BDH models on FineWeb-Edu:

| Recipe | Model | Context | Tokens | H100 time | Status |
|---|---|---|---|---|---|
| [1. Quick](#recipe-1-quick-d256) | D=256, about 34M parameters, 16K vocabulary | 512 | about 0.65B | **about 64 min (measured)** | Done: 1.1739 bpb, probe 14/50 |
| [2. Standard](#recipe-2-standard-d512-2048-token-context) | D=512, about 134M parameters, 32K vocabulary | **2048** | about 2.3B | about 14–17 h (estimate) | Planned |

Also on this page: an [experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe) on mixing in Wikipedia (no gain at 34M), a [comparison of BDH with a GPT](#experiment-bdh-vs-gpt-on-the-quick-recipe) trained the same way (the GPT wins at both equal parameters and equal compute), and [how to improve fact recall](#improving-fact-recall).

## Why FineWeb-Edu

- **Educational web pages** (`HuggingFaceFW/fineweb-edu`, 10B-token sample): explanations, textbooks, course material. More reasoning-style text per token than Wikipedia.
- **Plenty of data:** 14 parquet shards, about 700M tokens each. The model sees each token about once, so dropout isn't needed.
- **Long documents:** many are longer than 512 tokens, so there's real distant context for a longer window to learn from. TinyStories doesn't have this.
- **Better fact recall than Wikipedia at small sizes:** recipe 1 scored 14/50 on `probe.py`, against 6/50 for a Wikipedia model of the same size and tokens.

**Settings shared by both recipes:** `--n-layer 6 --n-head 4 --mlp-mult 128` (the defaults and the paper's settings; `--n-embd` is the size knob), `--dropout 0.0`, 32K tokens per step, `--weight-decay 0.1`, `--eval-freq 1000`.

**Comparing results:**
- **Bits per byte is comparable across tokenizers, not across datasets.** FineWeb-Edu's about 1.17 is higher than Wikipedia's about 1.0 because web text is more varied, not because the model is worse.
- **Validation bpb also changes with the block size and the data mix,** so compare checkpoints trained under the same settings.
- **For knowledge, use `probe.py`:** `python probe.py CHECKPOINT ... --verbose` compares checkpoints on the same 50 facts.

---

## Recipe 1: Quick (D=256)

A small model trained in about an hour. It matches the size of the Wikipedia model from the [quick start](README.md#quick-start), so comparing the two shows the effect of the data alone.

| Setting | Value |
|---|---|
| `--n-embd` | 256 |
| Vocabulary | 16384 (SentencePiece) |
| Data | 1 shard of FineWeb-Edu |
| `--block-size` | 512 |
| `--batch-size` / `--grad-accum` | 64 / 2 |
| `--max-iters` | 20,000 |
| `--lr` / `--warmup-iters` | 1e-3 / 1000 |

### Commands

```
python prepare_data.py fineweb-edu --tokenizer sentencepiece --vocab-size 16384

python train.py --data-dir data/fineweb-edu_sp16384 --ckpt-dir checkpoints/fwe_d256 \
    --n-embd 256 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 2 \
    --max-iters 20000 --lr 1e-3 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

One shard is the default, so the folder name has no `_1shards` suffix. On an 80GB GPU, `--grad-accum 1` probably fits too, and is a little faster.

### Results

On an H100:

| | |
|---|---|
| Speed | A steady 172K tokens/s, 190 ms per step: **about 64 minutes** for 20,000 steps |
| Validation | **1.1739 bpb at step 20,000** (1.2136 at step 13,000). It fell faster over the last few thousand steps, as the learning rate decayed. Training loss matched validation, so no overfitting |
| Bytes per token | About 3.9 |

**Fact recall** (`probe.py`, 50 probes, greedy), against the Wikipedia model of the same size:

| Checkpoint | Data | Tokens trained | Probe score |
|---|---|---|---|
| `fwe_d256/best.pt`, step 20,000 (final) | FineWeb-Edu, 1 shard | about 655M | **14/50 (28%)** |
| `fwe_d256/best.pt`, step 15,000 | FineWeb-Edu, 1 shard | about 490M | 9/50 (18%) |
| `wiki_sp16384/best.pt`, step 40,000 | Wikipedia, 2 shards | about 655M | 6/50 (12%) |

**What the results show:**
- **More training still helps:** continuing to step 30,000 (about 1.4 passes over the shard) brought validation down to **1.1588 bpb**, the control arm of the [Wikipedia experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe). So 20,000 steps leaves the model short of what this data can teach it.
- **FineWeb-Edu is ahead,** 14 against 6 with the same model size and tokens. Scores move by up to about 5 between nearby checkpoints (see the [experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe)), so treat this as likely rather than certain. A likely reason is repetition: small models learn a fact only after seeing it many times. Educational pages restate common knowledge constantly, while Wikipedia mostly states each fact once.
- **The last 5,000 steps added 5 facts** (9 to 14), while the learning rate decayed from about 3e-4 to 1e-4. That would fit the annealing effect, where a model consolidates what it learned as the rate drops. But 4,000 more steps on the same data later lost 5 again (see the experiment), so this is within the probe's noise.
- **It knows the most-repeated facts:** major dates (1914, 1945, 1789), London and Rome as capitals, water and oxygen, the Earth orbiting the Sun, Darwin and evolution. It missed rarer ones: 1492, 1776, birth years, chemical symbols, currencies.
- **Direction matters:** "The capital of France is" gives Paris, but "Paris is the capital of" gives Croatia, and the same for Japan and Tokyo. Language models often learn a fact only in the direction it's usually written ("the reversal curse", Berglund et al., 2023).
- **The Wikipedia model learned Wikipedia's templates:** its answers ("the municipality of…", "the province of…") copy the many short articles about small places. It learned the format more than the facts.
- **Text quality:** with `--top-k 20 --temperature 0.8`, fluent, article-like text with headings, but it drifts off topic ("Photosynthesis is the process" turned into an article about diet). A 34M model learns grammar and style before meaning.

---

## Experiment: a Wikipedia phase on the quick recipe

Status: **done.** Result: no measurable gain in facts from Wikipedia at 34M parameters, and a small cost in FineWeb-Edu quality. Recipe 2 keeps stage 2 on FineWeb-Edu only.

**Question:** does a final training phase on a mix of FineWeb-Edu and Wikipedia add facts, beyond what extra training on FineWeb-Edu alone would? It's a cheap test of the [optional Wikipedia mix in recipe 2's stage 2](#option-stage-2-with-wikipedia), and the first real run of dataset mixing in `train.py`.

### Design

Both arms continue recipe 1's finished model from step 20,000 to **step 30,000**, with the same settings. Only the data differs:

| Arm | Data | Checkpoint folder |
|---|---|---|
| **Mix** | 50% FineWeb-Edu, 50% Wikipedia | `checkpoints/fwe_d256_mix`, starting from a copy of the base checkpoint |
| **Control** | FineWeb-Edu only | `checkpoints/fwe_d256`: the base run itself, continued |

- **Why a control:** 10,000 more steps adds about 330M tokens. Any gain could come from the extra training alone. The difference between the arms is the effect of Wikipedia.
- **Wikipedia tokens:** the mix arm draws about 165M, less than one pass over 2 shards.
- **The base checkpoint is kept as a snapshot:** continuing `fwe_d256` overwrites its `latest.pt`, and its `best.pt` whenever validation improves. Recipe 1's final model stays available as `checkpoints/fwe_d256/step0020000_bpb1.1734.pt`.
- **No warmup on resume:** `train.py` computes the learning rate from the step number, and warmup only covers the first `--warmup-iters` steps of the whole run. A resumed run jumps straight to the cosine value for its `--max-iters`. The optimizer state is restored, so the jump is tolerable.
- **The arms' learning-rate paths differ.** Both end at 1e-4 at step 30,000, but the mix arm ran in two parts, and the control in one:

  | Arm | Steps 20,000–24,000 | Steps 24,000–30,000 |
  |---|---|---|
  | Mix | Jumps to about 1.7e-4, decays to 1e-4 (round 1, `--max-iters 24000`) | Jumps to about 1.9e-4, decays to 1e-4 (round 2, `--max-iters 30000`) |
  | Control | Jumps to about 3.4e-4, then decays to 1e-4 by step 30,000 in one run | |

  The control trains at higher rates for longer. Keep that in mind when comparing: part of any difference may come from the schedule, not the data.

### Commands

**1. Prepare Wikipedia with recipe 1's tokenizer.** Datasets in a mix must share one tokenizer:

```
python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece \
    --tokenizer-model data/fineweb-edu_sp16384/tokenizer.model --name wikipedia_2shards_fwe16k
```

**2. Start the mix arm from a copy of the base checkpoint:**

```
mkdir -p checkpoints/fwe_d256_mix
cp checkpoints/fwe_d256/latest.pt checkpoints/fwe_d256_mix/latest.pt
```

**3. Mix arm**, to step 30,000. Round 1 ran this with `--max-iters 24000` first; round 2 resumes from there:

```
python train.py --data-dir data/fineweb-edu_sp16384 data/wikipedia_2shards_fwe16k \
    --data-weights 0.5 0.5 --reset-best --ckpt-dir checkpoints/fwe_d256_mix \
    --n-embd 256 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 2 \
    --max-iters 30000 --lr 1e-3 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

`--reset-best` makes the mix arm's `best.pt` follow its own validation, which measures the mix.

**4. Control arm:** continue the base run to step 30,000:

```
python train.py --data-dir data/fineweb-edu_sp16384 --ckpt-dir checkpoints/fwe_d256 \
    --n-embd 256 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 2 \
    --max-iters 30000 --lr 1e-3 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

**5. Compare** the base model and both arms:

```
python probe.py checkpoints/fwe_d256/step0020000_bpb1.1734.pt \
    checkpoints/fwe_d256/latest.pt checkpoints/fwe_d256_mix/latest.pt --verbose
```

### What to look at

| Measure | Where | What it tells you |
|---|---|---|
| Probe score, mix vs. control | `probe.py` | **The main result:** facts added by Wikipedia beyond extra training |
| FineWeb-Edu validation bpb, mix vs. control | Each arm's eval lines (the mix arm prints one line per dataset) | The cost: how much general text quality the mix gives up. Both arms use the same validation batches, so the numbers compare directly. Also the low-noise check of whether the control improved |
| Wikipedia validation bpb | The mix arm's eval lines | Whether the model adapts to Wikipedia; it should fall steadily |
| Which facts changed | `--verbose` output | Whether Wikipedia adds rarer facts, or only shifts which borderline ones are right |

### Deciding

With 50 probes, differences of up to about 3 facts are within noise, and round 1 suggests the noise between checkpoints may be larger (see below).

| Outcome | For recipe 2 |
|---|---|
| Mix beats control by **4 or more** facts at step 30,000, with FineWeb-Edu bpb at most about 0.01 worse | Use the Wikipedia mix in stage 2 |
| Within **3 facts** | No clear effect at this scale. Keep stage 2 on FineWeb-Edu only, or use a smaller share such as 25% Wikipedia |
| Mix **worse** than control | Keep stage 2 on FineWeb-Edu only |

A 34M model has little spare capacity, so new facts may displace old ones. A null result here doesn't rule out a gain at 134M.

### Experiment results

**Round 1** (4,000 steps, to step 24,000). The control ran in a separate copy, `fwe_d256_cont`, which round 2 no longer needs:

| Checkpoint | Probe score | FineWeb-Edu val bpb | Wikipedia val bpb |
|---|---|---|---|
| Base, step 20,000 | 14/50 | 1.1739 | – |
| Control, step 24,000 | 9/50 | not recorded | – |
| Mix, step 24,000 | 12/50 | not recorded | not recorded |

**Reading round 1:**
- **Both arms fell below the base model,** the control by 5 facts with no change of data. That points to probe noise rather than either data choice: many facts sit right at the edge (the correct answer only slightly more likely than a wrong one), so small weight changes flip them. Small models also keep forgetting and relearning rarely seen facts as training continues ("forgetting events", Toneva et al., 2019).
- **Mix against control, 12 against 9,** is within noise: no evidence for or against Wikipedia.
- **4,000 steps was short for Wikipedia to add facts:** the mix arm saw about 65M Wikipedia tokens, so most facts in it appeared once or not at all. Round 2 extends both arms to 10,000 steps.

**Round 2** (to step 30,000):

| Checkpoint | Probe score | FineWeb-Edu val bpb | Wikipedia val bpb |
|---|---|---|---|
| Base, step 20,000 | 14/50 | 1.1739 | – |
| Control, step 30,000 | **12/50** | **1.1588** | – |
| Mix, step 30,000 | **11/50** | **1.1745** | 1.2127 |

**Reading round 2:**
- **Facts: no difference.** Mix against control is 11 against 12, well within noise. Both stay below the base model's 14, as in round 1.
- **Cost: about 0.016 bpb on FineWeb-Edu.** The control improved from 1.1739 to 1.1588 with 10,000 more steps on FineWeb-Edu. The mix arm stayed at 1.1745, since half its steps went to Wikipedia. That's beyond the 0.01 limit in the decision rule.
- **Wikipedia was barely absorbed:** the mix arm's Wikipedia bpb is 1.21, against about 1.0 for the Wikipedia-only model on the same validation text, after about 165M Wikipedia tokens.
- **The probe misses real improvement:** the control's validation bpb improved by 0.015, yet its probe score fell from 14 to 12. At this size, the 50-probe greedy score is too noisy to track small changes; validation bpb is the reliable signal.

**Conclusion:** by the decision rule, recipe 2's stage 2 stays on FineWeb-Edu only. A 34M model has little spare capacity, so this doesn't rule out a gain at 134M, but there's no evidence for one either. A less noisy probe, scoring the likelihood of the right answer rather than greedy matches, would be needed to measure small differences in facts.

---

## Experiment: BDH vs. GPT on the quick recipe

Status: **done.** The GPT beat BDH in both comparisons: 0.031 lower validation bpb at the same parameter count while training 5× faster, and 0.106 lower at about the same compute and training time.

**Question:** how does BDH compare with a standard transformer trained the same way? The BDH paper reports that BDH matches GPT-2-style transformers at the same parameter count. This tests that claim on your data, and also against a transformer given the same compute.

### Design

Each GPT is a Llama-style transformer ([`--arch gpt`](train.md#gpt-baseline---arch-gpt)) trained exactly like recipe 1: the same data and tokenizer, 20,000 steps of 32K tokens, the same schedule and validation batches. Only the model differs:

| Run | Model | Parameters | Compute per token (estimate) | Checkpoint folder |
|---|---|---|---|---|
| BDH (recipe 1) | D=256, 6 shared layers | 33.6M | about 260M multiply-adds | `checkpoints/fwe_d256` |
| **GPT, equal parameters** | 8 layers, D=512, 8 heads | 34.1M | about 40M | `checkpoints/gpt_d512` |
| **GPT, equal compute** | 16 layers, D=1024, 16 heads | 219M | about 240M | `checkpoints/gpt_d1024` |

- **Equal parameters** is the paper's comparison. A transformer of this size does about 7× less computation per token, so it should train several times faster.
- **Equal compute** gives the transformer as much computation per token as BDH. It's a much larger model on the same 655M tokens, only about 3 tokens per parameter, far below the usual 20. So it's undertrained by design: that's what the same compute budget buys a transformer here.
- **Validation bpb compares directly:** same tokenizer, same validation text, same batches.

### Commands

The data is recipe 1's. **GPT, equal parameters:**

```
python train.py --arch gpt --data-dir data/fineweb-edu_sp16384 --ckpt-dir checkpoints/gpt_d512 \
    --n-layer 8 --n-embd 512 --n-head 8 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 1 \
    --max-iters 20000 --lr 1e-3 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

**GPT, equal compute:**

```
python train.py --arch gpt --data-dir data/fineweb-edu_sp16384 --ckpt-dir checkpoints/gpt_d1024 \
    --n-layer 16 --n-embd 1024 --n-head 16 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 2 \
    --max-iters 20000 --lr 6e-4 --warmup-iters 1000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000
```

A transformer of 34M parameters needs far less activation memory than BDH, so `--grad-accum 1` should fit; the 219M model may need 2. `--lr 6e-4` for the larger model follows the usual practice of lower rates for larger models.

**Compare:**

```
python probe.py checkpoints/fwe_d256/step0020000_bpb1.1734.pt \
    checkpoints/gpt_d512/latest.pt checkpoints/gpt_d1024/latest.pt --verbose
python inference.py "Photosynthesis is the process" --checkpoint checkpoints/gpt_d512/latest.pt \
    --max-new-tokens 300 --top-k 20 --temperature 0.8
```

### What to look at

| Measure | Where | What it tells you |
|---|---|---|
| Validation bpb at step 20,000 | Eval lines | **The main result:** quality at the same parameters, or the same compute, and the same tokens |
| Validation bpb over time | Eval lines every 1,000 steps | Whether one model learns faster early, or keeps improving longer |
| Tokens/s and total time | Log lines and the final "Trained … in …" line | Training cost. The equal-parameter GPT should be several times faster |
| Probe score | `probe.py` | Facts. Differences of up to about 5 are within noise (see the Wikipedia experiment) |
| Samples | `inference.py` | Coherence and staying on topic, judged by reading |

Generation speed in `inference.py` isn't a fair comparison yet: the GPT has no KV cache, so it re-reads the context for every token, like BDH's default method. BDH's recurrent mode has no GPT equivalent here.

### Results

| Run | Parameters | Val bpb, step 20,000 | Probe | Tokens/s | Training time |
|---|---|---|---|---|---|
| BDH, recipe 1 | 33.6M | 1.1739 | 14/50 | 172K | about 64 min |
| **GPT, equal parameters** | 34.1M | **1.1434** | 13/50 | **871K** | **about 12.5 min** |
| **GPT, equal compute** | 219M | **1.0682** | **16/50** | 196K | about 56 min |

**Reading the equal-parameter result:**
- **The GPT is better: 0.031 bpb lower** at the same parameters, tokens, schedule and validation batches. It even beats BDH trained 50% longer (1.1588 bpb at step 30,000, from the [Wikipedia experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe)'s control).
- **And 5× faster to train:** 871K tokens/s against 172K, 12.5 minutes against about 64. That fits the estimate of about 7× less computation per token, with BDH's large element-wise operations adding overhead.
- **Facts: no difference.** 13 against 14 is within the probe's noise.
- **Text:** the sample after training is the usual "To be or " prompt with `--top-k 3`, and loops like BDH's. It says little about either model.
- **So at this size, on this data, the paper's claim doesn't hold up here:** a modern transformer with the same parameter count reached lower loss, at a fifth of the training cost.

**Reading the equal-compute result:**
- **The GPT is far better: 0.106 bpb lower** than BDH, with about the same computation per token and slightly less training time (about 56 minutes against 64).
- **Despite being undertrained:** 219M parameters on 655M tokens is about 3 tokens per parameter, far below the usual 20. A transformer still turns the same compute into much lower loss than BDH.
- **Facts: the best score so far,** 16/50. Two more than BDH is within the probe's noise on its own, but it fits the much lower loss and the larger model's capacity.
- **Speed per token is about the same as BDH's** (196K tokens/s against 172K), confirming the estimate that BDH does as much computation per token as a 16-layer, D=1024 transformer.
- **Text:** the sample after training uses the "To be or " prompt with `--top-k 3`, which led it into a list of numbers. It says little.

**Overall:** at this size, on this data, BDH loses to a modern transformer both at equal parameters and at equal compute. BDH's remaining advantage is in inference: constant memory and time per generated token in recurrent mode.

**Caveats:**
- **One run each, untuned:** both used recipe 1's settings (learning rate 1e-3, warmup 1,000, weight decay 0.1), chosen for BDH. Neither was tuned, so the gap could shrink or grow with tuning. A seed-to-seed difference in validation bpb is usually below 0.005, so 0.031 is unlikely to be noise.
- **A modern transformer, not GPT-2:** the paper compares with GPT-2-style models. `gpt.py` is Llama-style (RMSNorm, RoPE, SwiGLU), which is typically a few percent better in loss than GPT-2's design.
- **One size:** the paper reports results from about 10M to 1B parameters. The comparison could differ at other sizes.
- **Not a comparison of inference:** BDH's recurrent mode gives constant memory and time per generated token; the GPT's grows with context. For long contexts, that can still favor BDH, at a cost in quality per parameter and in training compute.

---

## Recipe 2: Standard (D=512, 2048-token context)

A D=512 model of about 134M parameters, trained on about 2.3B tokens in two stages, ending with a **2048-token context window**.

Status: **planned.** Times are estimates scaled from recipe 1's measured speed, not measurements. Check them before committing to the full run (see [Before the long run](#before-the-long-run)).

### The method

1. **Stage 1: train at 512 tokens** for about 85% of the steps. Short blocks are cheap, and the model learns the language there.
2. **Stage 2: continue at 2048 tokens** for the last 15%, while the learning rate decays. The model learns to use the longer context. The [Wikipedia experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe) found no gain from mixing in Wikipedia at 34M parameters, so stage 2 stays on FineWeb-Edu.

Most long-context models are trained this way. Training at 2048 from the start works too, but costs much more for little gain:

| Block size | Training cost per token (512 = 1×) | 2.3B tokens, all at this length |
|---|---|---|
| 512 | 1× | about 12–16 h |
| 1024 | about 1.25× | about 15–20 h |
| 2048 | about 1.75× | about 21–28 h |
| 2048 with `--attn-chunk 256` | about 1.4× | about 17–22 h |
| **This recipe: 60K steps at 512, then 10K at 2048 with `--attn-chunk 256`** | | **about 14–17 h** |

Attention cost grows with block length; the rest of the model's cost doesn't. At D=512, attention is about a quarter of the cost at 512 tokens, and more than half at 2048. [Chunked attention](train.md#chunked-attention-for-long-blocks---attn-chunk) (`--attn-chunk`) computes the same model with a cost that grows linearly with block length. At D=512 it pays off above about 1,300 tokens, so it's used in stage 2 only.

### Settings

| Setting | Stage 1 | Stage 2 | Why |
|---|---|---|---|
| `--n-embd` | 512 | 512 | 4× recipe 1's size, and still overnight on one H100 |
| Vocabulary | 32768 | 32768 | Web text is varied, so a larger vocabulary puts more text in each token, and each block holds more context. The extra embedding parameters are cheap at D=512 |
| Parameters | about 134M | | 3 · 128 · 512² + 2 · 32768 · 512 |
| `--block-size` | 512 | 2048 | |
| `--batch-size` | 64 | 16 | 32K tokens per step in both stages |
| `--grad-accum` | 4 | 4 | BDH's sparse activations are large, so a full batch doesn't fit in 80GB. Micro-batches are 8K tokens in both stages |
| `--attn-chunk` | off | 256 | Faster only above about 1,300 tokens at D=512 |
| Steps | 0–60,000 | 60,000–70,000 | About 2.3B tokens in total, close to 20 tokens per parameter |
| `--lr` | 6e-4 | 6e-4 | Lower than D=256's 1e-3: larger models are less stable at the same rate |
| `--warmup-iters` | 2000 | 2000 | |

### Commands

**1. Prepare the data:** 4 shards, about 2.8B tokens, with a new 32K-vocabulary tokenizer:

```
python prepare_data.py fineweb-edu --shards 4 --tokenizer sentencepiece --vocab-size 32768
```

Check the folder name it prints; the commands below assume `data/fineweb-edu_4shards_sp32768`.

**2. Stage 1:** 60,000 steps at 512 tokens:

```
python train.py --data-dir data/fineweb-edu_4shards_sp32768 --ckpt-dir checkpoints/fwe_d512 \
    --n-embd 512 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 512 --batch-size 64 --grad-accum 4 \
    --max-iters 60000 --lr 6e-4 --warmup-iters 2000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 10000
```

**3. Stage 2:** resume, and continue to 70,000 steps at 2048 tokens:

```
python train.py --data-dir data/fineweb-edu_4shards_sp32768 --ckpt-dir checkpoints/fwe_d512 \
    --n-embd 512 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 2048 --batch-size 16 --grad-accum 4 --attn-chunk 256 \
    --max-iters 70000 --lr 6e-4 --warmup-iters 2000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000 --reset-best
```

It resumes from `checkpoints/fwe_d512/latest.pt` at step 60,000. Keep the step-60,000 snapshot from stage 1 for comparisons.

#### Option: stage 2 with Wikipedia

Not recommended at present: the [experiment](#experiment-a-wikipedia-phase-on-the-quick-recipe) at 34M parameters found no gain in facts and a cost of about 0.016 bpb on FineWeb-Edu. It's kept here for testing at larger sizes, **instead of** the stage 2 command above. Wikipedia is knowledge-dense, and mixing it in while the learning rate decays can improve fact recall (see [Improving fact recall](#improving-fact-recall)).

Prepare Wikipedia with this recipe's tokenizer, any time before stage 2:

```
python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece \
    --tokenizer-model data/fineweb-edu_4shards_sp32768/tokenizer.model --name wikipedia_2shards_fwe
```

Stage 2 with a 50/50 mix:

```
python train.py --data-dir data/fineweb-edu_4shards_sp32768 data/wikipedia_2shards_fwe \
    --data-weights 0.5 0.5 --ckpt-dir checkpoints/fwe_d512 \
    --n-embd 512 --n-head 4 --mlp-mult 128 --n-layer 6 --dropout 0.0 \
    --block-size 2048 --batch-size 16 --grad-accum 4 --attn-chunk 256 \
    --max-iters 70000 --lr 6e-4 --warmup-iters 2000 --weight-decay 0.1 \
    --eval-freq 1000 --snapshot-freq 5000 --reset-best
```

- **Repetition:** stage 2 draws about 165M Wikipedia tokens, less than one pass over 2 shards.
- **Evaluation takes about twice as long,** because each dataset is evaluated separately. The log shows both losses, so you can see whether general text suffers while Wikipedia improves.

### What happens at the switch to stage 2

- **Changing the block size on resume works.** `train.py` checks the model options and the tokenizer when it resumes, not the block size. Stage 2 checkpoints record 2048, so `inference.py`, `chat.py` and `finetune.py` use a 2048-token window automatically.
- **`--reset-best`:** with more context, validation loss is lower, and with a data mix it measures different text. Either way it isn't comparable with stage 1's, so `best.pt` restarts from the first stage-2 evaluation.
- **The learning rate ticks up slightly.** Stage 1's schedule ends at 6e-5. Stage 2 resumes partway through a 70,000-step schedule, at about 9e-5, and decays back to 6e-5. A small bump like this is harmless.
- **Memory stays about the same:** micro-batches are 8K tokens in both stages, and with `--attn-chunk`, no `block × block` score matrices are stored.
- **`--attn-chunk` changes only how attention is computed,** not the model.

### Before the long run

1. **Measure the speed.** Run stage 1 for about 200 steps and read tok/s from the log. Stage 1 hours ≈ 1.97B tokens ÷ tok/s ÷ 3600; stage 2 is about 1.4× slower per token. Stop with Ctrl+C; running the same command again resumes.
2. **Check memory.** If either stage runs out of memory, double `--grad-accum`. Tokens per step stay the same, so results don't change.
3. **Check chunked attention on your GPU:** `python test_chunked.py --bench --n-embd 512` checks it against full attention, then times training steps at several block sizes. Use the fastest chunk size at 2048, or drop `--attn-chunk` if full attention is faster.
4. **Try stage 2 briefly** before the real switch: copy an early `latest.pt` into a scratch `--ckpt-dir` and run the stage 2 command for a few dozen steps.

### During training

- **Validation bpb should still be falling at step 60,000.** If it flattens early, the model is too small for the data; see the larger variant below.
- **Loss spikes** that don't recover: resume with a lower `--lr`, e.g. 4e-4. Resuming applies the new learning rate.
- **Snapshots** every 10,000 steps (5,000 in stage 2) keep fallback points, named by bits per byte.

### After training

- **Fact recall:** `python probe.py checkpoints/fwe_d512/best.pt checkpoints/fwe_d256/best.pt --verbose`.
- **Long-context use:** whether the model actually uses all 2,048 tokens is measured by the length evaluation in phase 1 of [long_context_plan.md](long_context_plan.md).
- **Chat:** `finetune.py --base checkpoints/fwe_d512/best.pt`. It takes the block size from the base checkpoint, so fine-tuning keeps the 2048 window. See [chat.md](chat.md).
- **Recurrent inference:** with the sliding window, the state grows with the window. At D=512 with 2048 tokens, it's about 4GB: about 0.8GB for S and about 3.2GB for the ring buffer. Saved state files are that size too. Speed per token stays constant. See [recurrent.md](recurrent.md).

### Extensions

- **Longer windows:** for 4K–8K tokens, continue from stage 2 with `--block-size 4096` (then 8192), `--attn-chunk 256`, and `--batch-size 8` (then 4) to keep 32K tokens per step. Chunked attention keeps the attention cost growing linearly. That's phase 5 of [long_context_plan.md](long_context_plan.md).
- **Larger variant:** `--n-embd 768` (about 276M parameters), 7–8 shards (about 5B tokens), `--grad-accum 8` at 512 tokens, `--lr 4e-4`, and 150,000 steps, e.g. 130,000 at 512 then 20,000 at 2048. About 3 days on an H100 (estimate). Otherwise the same commands.

### What to expect

A 134M-parameter model gives fluent, on-topic educational prose and some basic facts. It won't give reliable answers or multi-step reasoning: those need much larger models and more data.

---

## Improving fact recall

How to make a model remember more facts, as measured by `probe.py`, roughly in order of impact. The findings on knowledge capacity come from studies of transformers (Allen-Zhu & Li, "Physics of Language Models", part 3); they're not measured for BDH.

1. **A larger model: the biggest lever.** Language models store at most about **2 bits of knowledge per parameter**, and they spend capacity on grammar and style first. A 34M model has room for only a limited number of facts; recipe 2's 134M model has about 4× the room.
2. **More exposures per fact.** A model needs on the order of **hundreds of exposures** to a fact to store it near that capacity.
   - **Train well past 20 tokens per parameter.** 20 is the compute-optimal point for loss, not for knowledge, which keeps improving with longer training. Small open models are trained on 100–1000+ tokens per parameter.
   - **Repeating knowledge-dense data is fine:** 2–4 passes over Wikipedia cost little compared with fresh data.
   - **Data that restates common facts helps small models,** a likely reason FineWeb-Edu beat Wikipedia in recipe 1.
3. **A better data mix.**
   - **Mix general text with knowledge-dense text,** e.g. FineWeb-Edu with Wikipedia: FineWeb-Edu repeats common facts, and Wikipedia covers many more. `train.py --data-dir A B --data-weights 0.5 0.5`; see [Mixing datasets](train.md#mixing-datasets).
   - **Anneal on knowledge-dense data:** spend the last 10–20% of training, while the learning rate decays, mostly on high-quality text such as Wikipedia. Recent small models are trained this way. Recipe 1's probe scores were too noisy to confirm this at 34M parameters.
   - **The same fact in several phrasings** helps a model retrieve it from different prompts. Facts seen in one wording or one direction are often stored but hard to retrieve.
4. **Chat fine-tuning (small effect).** `finetune.py` teaches the model to answer rather than continue text. It helps express facts the model already knows, but doesn't add new ones.
5. **Changes to the probe: better measurement, not more knowledge.** Few-shot prompts (two or three solved examples before each question) and multiple-choice scoring (the likelihood of the right answer against wrong ones) usually raise scores and detect partial knowledge. Neither is implemented in `probe.py` yet; keep the current probe as the baseline whatever is added.
6. **Retrieval, for reliable facts in practice.** Put the relevant passage in the model's context instead of relying on its memory. Even small models answer well when the answer is in front of them. That measures reading, not memory, so it's a different score.
