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
| `wikipedia` | English Wikipedia, 2023-11-01 dump (`wikimedia/wikipedia` on Hugging Face): 41 parquet shards of about 330–420MB | The last `--val-fraction` (default 0.005) of the joined text |
| `fineweb-edu` | FineWeb-Edu 10B-token sample (`HuggingFaceFW/fineweb-edu`): educational web pages, 14 parquet shards of about 2.15GB | The last `--val-fraction` (default 0.005) of the joined text |
| `text` | Any text file given with `--input` | The last `--val-fraction` of the file |

Downloads are cached in `data/raw/<dataset>/`. Preparing the same dataset again, for example with a different tokenizer, reuses the cache. Splits and truncation always fall on line boundaries.

### Wikipedia and FineWeb-Edu

These are knowledge-heavy datasets: real encyclopedia and web text with varied vocabulary and many facts. They're much harder to model than TinyStories.

- **Shards:** `--shards N` downloads the first N shards. The default is 1, and the output folder name gets a `_<N>shards` suffix when N is above 1.
- **Document format:** documents are joined into one text file with an `<|endoftext|>` line after each, the same format as TinyStories. Wikipedia articles start with their title and a blank line.
- **Caching:** the parquet shards and the joined text (`text_<N>shards.txt`) are both kept in `data/raw/<dataset>/`. Increasing `--shards` later reuses the shards already downloaded.
- **Dependency:** reading parquet needs `pyarrow` (`pip install pyarrow`, included in `requirements.txt`).

## Tokenizers

| `--tokenizer` | Vocab | Stored as | Notes |
|---|---|---|---|
| `bytes` (default) | 256 | uint8 | Each UTF-8 byte is one token. `train.bin` is the raw text. |
| `sentencepiece` | `--vocab-size` (default 4096) | uint16 | BPE trained on the training split only. Special tokens stay whole, and unknown characters fall back to bytes. Decoding reproduces the text exactly. |

A SentencePiece tokenizer is trained on up to `--tokenizer-sample-lines` lines sampled from the training split, never on validation text. It is saved as `data/<name>/tokenizer.model`.

## Options

| Option | Default | Description |
|---|---|---|
| `dataset` | required | `tinystories`, `shakespeare`, `wikipedia`, `fineweb-edu` or `text` |
| `--input FILE` | none | Text file to use; required for `text` |
| `--name NAME` | dataset name, plus `_<N>shards` and `_sp<vocab>` where they apply | Output folder under `--out-dir` |
| `--out-dir DIR` | `data` | Parent folder for prepared datasets; downloads go to `<out-dir>/raw` |
| `--val-fraction F` | 0.1, or 0.005 for `wikipedia`/`fineweb-edu` | Share of the text held out for validation (all datasets except `tinystories`) |
| `--shards N` | 1 | Parquet shards to download (`wikipedia`: up to 41, `fineweb-edu`: up to 14) |
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

# Wikipedia, first 2 shards, 16K BPE vocabulary (varied text needs a larger vocab than TinyStories)
python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece --vocab-size 16384

# FineWeb-Edu, one shard (about 2.15GB download)
python prepare_data.py fineweb-edu --tokenizer sentencepiece --vocab-size 16384

# Your own corpus, 5% held out for validation
python prepare_data.py text --input corpus.txt --val-fraction 0.05

# Encode another dataset with an existing tokenizer, so a model can be trained or evaluated on both
python prepare_data.py text --input other.txt --tokenizer sentencepiece \
    --tokenizer-model data/tinystories_sp4096/tokenizer.model --name other_sp4096

# Wikipedia with FineWeb-Edu's tokenizer, to mix the two in training (train.py --data-dir A B)
python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece \
    --tokenizer-model data/fineweb-edu_4shards_sp32768/tokenizer.model --name wikipedia_2shards_fwe
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

- **Choosing a vocab size:** a small vocabulary (2K–8K) suits small models on a single simple dataset like TinyStories. For Wikipedia or web text, use 16K–32K, because rarer words would otherwise be split into many small pieces. The embedding and output layers add `2 × vocab × n_embd` parameters: 4096 adds about 2M at the default `n_embd=256`, 16384 about 8M, and GPT-2's 50K about 26M.
- **Tiny Shakespeare:** stay with `bytes`. The dataset is too small to train a good tokenizer.
- **What to expect from knowledge-heavy data:** a small model learns fluent, encyclopedia-*sounding* text, but it can't store many facts. It will drift off topic and make up plausible names, dates and numbers. Expect a clearly higher bits per byte than on TinyStories. That's the point of comparison, not a sign that something is broken.
- **Disk space:** TinyStories needs about 2.2GB for the cached download. A byte-level dataset adds another 2.2GB, a 4096-vocab SentencePiece dataset roughly 1.2GB. For Wikipedia and FineWeb-Edu, the cache holds both the parquet shards and the joined text, which is larger than the parquet because parquet is compressed.
