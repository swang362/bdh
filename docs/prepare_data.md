# prepare_data.py

`prepare_data.py` turns a text dataset into `train.bin` and `val.bin` token files for `train.py --data-dir`. It can also train a tokenizer.

```
python prepare_data.py <dataset> [options]
```

Output goes to `data/<name>/`. At the end, the script prints the matching `train.py` command.

## Datasets

| Dataset | Source | Validation split |
|---|---|---|
| `tinystories` | TinyStories V2 (GPT-4), about 2.2GB of simple English stories separated by `<|endoftext|>` | The dataset's own `valid` file, about 22MB |
| `shakespeare` | Tiny Shakespeare, about 1.1MB | The last `--val-fraction` of the file |
| `text` | Any text file given with `--input` | The last `--val-fraction` of the file |

Downloads are cached in `data/raw/<dataset>/`. Preparing the same dataset again, for example with a different tokenizer, reuses the cache. Splits and truncation always fall on line boundaries.

## Tokenizers

| `--tokenizer` | Vocab | Stored as | Notes |
|---|---|---|---|
| `bytes` (default) | 256 | uint8 | Each UTF-8 byte is one token. `train.bin` is the raw text. |
| `sentencepiece` | `--vocab-size` (default 4096) | uint16 | BPE trained on the training split only. Special tokens stay whole, and unknown characters fall back to bytes. Decoding reproduces the text exactly. |

A SentencePiece tokenizer is trained on up to `--tokenizer-sample-lines` lines sampled from the training split, never on validation text. It is saved as `data/<name>/tokenizer.model`.

## Options

| Option | Default | Description |
|---|---|---|
| `dataset` | required | `tinystories`, `shakespeare` or `text` |
| `--input FILE` | none | Text file to use; required for `text` |
| `--name NAME` | dataset name, plus `_sp<vocab>` for sentencepiece | Output folder under `--out-dir` |
| `--out-dir DIR` | `data` | Parent folder for prepared datasets; downloads go to `<out-dir>/raw` |
| `--val-fraction F` | 0.1 | Share of the file held out for validation (`shakespeare` and `text` only) |
| `--max-train-bytes N` | none | Truncate the training split to about N bytes. Underscores are allowed, e.g. `100_000_000` |
| `--tokenizer` | `bytes` | `bytes` or `sentencepiece` |
| `--vocab-size N` | 4096 | SentencePiece vocabulary size |
| `--special-tokens T ...` | `<|endoftext|>` | Tokens that are never split. Pass the flag with nothing after it for none |
| `--tokenizer-model FILE` | none | Reuse an existing SentencePiece `.model` instead of training one |
| `--tokenizer-sample-lines N` | 2,000,000 | Lines sampled to train the tokenizer |

## Examples

```
# TinyStories, byte-level
python prepare_data.py tinystories

# TinyStories with an 8K BPE vocabulary
python prepare_data.py tinystories --tokenizer sentencepiece --vocab-size 8192

# A 100MB subset for quick experiments (the full file is still downloaded and cached once)
python prepare_data.py tinystories --max-train-bytes 100_000_000 --name ts_100mb

# Your own corpus, 5% held out for validation
python prepare_data.py text --input corpus.txt --val-fraction 0.05

# Encode another dataset with an existing tokenizer, so a model can be trained or evaluated on both
python prepare_data.py text --input other.txt --tokenizer sentencepiece \
    --tokenizer-model data/tinystories_sp4096/tokenizer.model --name other_sp4096
```

## meta.json

`train.py` reads `meta.json` to choose the tokenizer, vocab size and data type:

```json
{
  "dataset": "tinystories",
  "tokenizer": "sentencepiece",
  "tokenizer_model": "tokenizer.model",
  "vocab_size": 4096,
  "dtype": "uint16",
  "special_tokens": ["<|endoftext|>"],
  "train_tokens": 600000000,
  "val_tokens": 6000000,
  "train_bytes": 2227753162,
  "val_bytes": 22502601,
  "bytes_per_token": 3.7
}
```

The token counts and `bytes_per_token` above are illustrative. The script prints the real values when it finishes.

Use `bytes_per_token` to compare losses across tokenizers: `bits per byte = loss / ln(2) / bytes_per_token`. `train.py` does this conversion for you.

## Tips

- **Choosing a vocab size:** a small vocabulary (2K–8K) suits small models on a single dataset. The embedding and output layers add `2 × vocab × n_embd` parameters: 4096 adds about 2M at the default `n_embd=256`, while GPT-2's 50K would add about 26M.
- **Tiny Shakespeare:** stay with `bytes`. The dataset is too small to train a good tokenizer.
- **Disk space:** TinyStories needs about 2.2GB for the cached download. A byte-level dataset adds another 2.2GB, a 4096-vocab SentencePiece dataset roughly 1.2GB.
