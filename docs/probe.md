# probe.py

`probe.py` measures **fact recall**. It completes 50 factual prompts with greedy decoding and checks each completion for an expected answer. The result is a simple accuracy score to track alongside validation bits per byte: across the snapshots of one run, between model sizes, or between architectures.

```
python probe.py <checkpoint or directory> [...] [options]
```

## Examples

```
# One checkpoint
python probe.py checkpoints/wiki_sp16384/best.pt

# Every checkpoint in a run, showing each completion: how facts are learned over training
python probe.py checkpoints/wiki_sp16384 --verbose

# Compare two runs and save the results
python probe.py checkpoints/wiki_sp16384 checkpoints/wiki_d384 --csv probe.csv

# Your own probes
python probe.py checkpoints/wiki_sp16384/best.pt --probes my_probes.json
```

## Options

| Option | Default | Description |
|---|---|---|
| `checkpoints` | required | Checkpoint files, or directories. Every `.pt` file in a directory is probed: snapshots, `best.pt` and `latest.pt` |
| `--probes FILE` | built-in list | JSON list of `{"prompt": "...", "answers": ["...", ...]}` |
| `--max-new-tokens N` | 10 | Tokens generated per probe |
| `--context-size N` | training block size from the checkpoint | Context window, as in `inference.py` |
| `--verbose` | off | Print every completion, marked `[OK]` or `[--]` |
| `--csv FILE` | none | Also write one row per checkpoint: checkpoint, step, correct, total, accuracy |
| `--cpu` | off | Run on the CPU |

## Output

```
checkpoints/wiki_sp16384/best.pt (step 36000): 11/50 correct (22.0%)
  [OK] The capital of France is -> 'the city of Paris.'  (expected: Paris)
  [--] World War II ended in -> '1940.'  (expected: 1945)
  ...

Summary (by training step):
  step    10000    4/50    8.0%  step0010000_bpb1.0812.pt
  step    20000    8/50   16.0%  step0020000_bpb1.0391.pt
  step    36000   11/50   22.0%  best.pt
```

The numbers above are illustrative.

## How scoring works

- **Greedy decoding** (top-k 1): the model's single most likely continuation, the same on every run.
- **Correct** means any accepted answer appears in the completion as a **whole word**, ignoring case. So `Au` doesn't count as a match inside "August", and `1945` doesn't match "19450".
- **Some probes accept several answers,** e.g. `Ulm` or `1879` for Einstein's birth. The whole completion is checked, so a model that names the answer a few words later still gets credit.
- **The completion stops at `<|endoftext|>`.**

## The built-in probes

There are 50 probes, covering capitals, dates of major events, birthplaces of well-known people, basic science, geography, languages and currencies.

- **Most are phrased the way Wikipedia writes,** e.g. "Paris is the capital of". A model that has only been pretrained continues text rather than answering questions.
- **A few are question-like,** e.g. "The capital of France is", to show the difference phrasing makes.

## Interpreting the score

- **It measures memorized knowledge, not language quality.** A model can write fluent Wikipedia-style text and still score low. That's typical for small models.
- **Frequent facts are learned first,** e.g. Paris and France. Specific numbers and less common facts need much larger models.
- **50 probes give a coarse score:** one probe is 2 percentage points. Treat differences of a few probes as noise, and compare trends across checkpoints.
- **Probes suit models trained on knowledge-heavy data** (`wikipedia`, `fineweb-edu`). A TinyStories model will score near zero, as expected.
