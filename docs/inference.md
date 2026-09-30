# inference.py

`inference.py` generates text from a checkpoint. By default it streams the output as it's generated and stops at `<|endoftext|>`. It then prints speed and resource usage.

```
python inference.py "<prompt>" [options]
```

The checkpoint provides the model config, the tokenizer and the training block size, so no other setup is needed.

## Options

| Option | Default | Description |
|---|---|---|
| `prompt` | required | Text to continue. It can contain `<|endoftext|>`, which is a single token with SentencePiece |
| `--checkpoint FILE` | `checkpoints/latest.pt` | Checkpoint or snapshot to load |
| `--max-new-tokens N` | 200 | Maximum tokens to generate |
| `--context-size N` | training block size from the checkpoint | How many recent tokens the model sees at each step. `0` means unlimited (not recommended, see below) |
| `--recurrent` | off | Generate with a fixed-size recurrent state instead of re-reading the context for every token. Much less compute per token, and constant speed however long the output. Identical output while prompt plus output fit in the context window. See [recurrent.md](recurrent.md) |
| `--temperature T` | 1.0 | Sampling temperature; lower is more predictable. Must be above 0 |
| `--top-k K` | 3 | Only sample from the K most likely tokens. `--top-k 1` always picks the most likely token (greedy) |
| `--seed N` | none | Makes the output reproducible |
| `--stream` / `--no-stream` | on | Print tokens as they're generated, or all at once at the end |
| `--stop-at-eot` / `--no-stop-at-eot` | on | Stop when the model generates `<|endoftext|>`. The marker itself isn't printed |
| `--cpu` | off | Run on the CPU even if a GPU is available |
| `--resources` / `--no-resources` | on | Print CPU, RAM and GPU usage after generation |

## Examples

```
python inference.py "Once upon a time"
python inference.py "Once upon a time" --checkpoint checkpoints/ts_sp4096/step0040000_bpb0.4123.pt
python inference.py "ROMEO:" --max-new-tokens 500 --temperature 0.8 --top-k 10 --seed 42
python inference.py "Once upon a time" --top-k 1          # greedy
python inference.py "The end.<|endoftext|>" --max-new-tokens 300   # start a fresh story
python inference.py "Once upon a time" --cpu --no-resources
python inference.py "Once upon a time" > story.txt        # the file gets only the generated text
```

## Output

The generated text goes to **stdout**. Status and statistics go to **stderr**, so redirecting stdout captures only the text:

```
Loaded checkpoints/latest.pt (step 40000) on cuda, sentencepiece tokenizer (vocab 4096), context 512
Once upon a time, there was a little girl named Lily...

Generated 187 tokens in 2.10s (89.0 tok/s), first token after 31ms, stopped at <|endoftext|>
Model: 27,262,976 params (104.0MB weights)
CPU: 1.95s CPU time (93% of one core), 8 torch threads
RAM: peak 1,204.3MB
GPU: NVIDIA GeForce RTX 4090, peak allocated 412.6MB, reserved 520.0MB of 24,563.5MB
```

The numbers above are illustrative.

- **Model:** the parameter count and the weights' memory in float32.
- **CPU time** can exceed wall time when torch uses several threads.
- **RAM** requires `psutil` on Windows. On Linux and macOS it uses the built-in `resource` module.
- **The GPU line** appears for CUDA only. Peak memory counts generation only, not checkpoint loading.

## Context window

BDH's attention adds up contributions from all previous tokens without normalization. Past the length used in training, those sums go beyond anything the model saw, and the output degrades into random tokens. So each step only feeds the model the most recent `--context-size` tokens, which defaults to the training `block_size` stored in the checkpoint.

- **Long outputs stay fluent,** but the model can't see anything older than the window. In long outputs it may forget names or earlier details.
- **Checkpoints from before `block_size` was saved** fall back to 512 and print a warning. Pass `--context-size` if you trained with a different `--block-size`.
- **Speed stays constant** per token, however long the output gets.
- **With `--recurrent`,** the same window applies, but each step costs far less: the model keeps a running memory instead of re-reading the window. Past the window there's one small difference in how older tokens are represented; [recurrent.md](recurrent.md) explains it.

## Tips

- **Temperature above 0:** `--temperature 0` fails with a division by zero. Use `--top-k 1` for predictable output.
- **Byte-level models:** stopping at `<|endoftext|>` works for them too. The marker is matched in the decoded text, and partial output is held back until it's clear whether the marker is being generated.
- **Old checkpoints** (from before tokenizer support) load as byte-level models.
