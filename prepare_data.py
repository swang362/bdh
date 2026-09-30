# Copyright Pathway Technology, Inc.

"""Prepare a dataset as train.bin / val.bin for train.py --data-dir.

The model is byte-level (vocab_size=256), so each .bin file is simply the raw
UTF-8 bytes of the text, read by train.py as a uint8 memmap. Files are
streamed in chunks, so multi-GB datasets never need to fit in memory.

Examples:
    python prepare_data.py tinystories
    python prepare_data.py tinystories --max-train-bytes 100_000_000
    python prepare_data.py shakespeare
    python prepare_data.py text --input my_corpus.txt --name my_corpus
"""

import argparse
import json
import os

import requests

CHUNK_SIZE = 16 * 1024 * 1024
ROOT = os.path.dirname(os.path.abspath(__file__))

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def download(url, dst, max_bytes=None):
    """Stream url to dst, stopping after max_bytes if given. Returns bytes written."""
    tmp = dst + ".part"
    written = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        if max_bytes is not None:
            total = min(total, max_bytes) if total else max_bytes
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if max_bytes is not None:
                    chunk = chunk[: max_bytes - written]
                f.write(chunk)
                written += len(chunk)
                pct = f" ({100 * written / total:.0f}%)" if total else ""
                print(f"\r  {os.path.basename(dst)}: {fmt_bytes(written)}{pct}", end="")
                if max_bytes is not None and written >= max_bytes:
                    break
    print()
    os.replace(tmp, dst)
    return written


def copy_range(src, dst, start, length):
    """Copy length bytes of src starting at start into dst. Returns bytes written."""
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fin.seek(start)
        while written < length:
            chunk = fin.read(min(CHUNK_SIZE, length - written))
            if not chunk:
                break
            fout.write(chunk)
            written += len(chunk)
    return written


def split_text_file(src, out_dir, val_fraction, max_train_bytes):
    """Split one text file by byte position: first part train, last part val."""
    size = os.path.getsize(src)
    n_val = int(size * val_fraction)
    n_train = size - n_val
    if max_train_bytes is not None:
        n_train = min(n_train, max_train_bytes)
    train = copy_range(src, os.path.join(out_dir, "train.bin"), 0, n_train)
    val = copy_range(src, os.path.join(out_dir, "val.bin"), size - n_val, n_val)
    return train, val


def prepare_tinystories(out_dir, args):
    # TinyStories ships its own train/valid split; stories are separated by <|endoftext|>
    train = download(
        TINYSTORIES_URL.format(split="train"),
        os.path.join(out_dir, "train.bin"),
        max_bytes=args.max_train_bytes,
    )
    val = download(
        TINYSTORIES_URL.format(split="valid"), os.path.join(out_dir, "val.bin")
    )
    return train, val


def prepare_shakespeare(out_dir, args):
    raw = os.path.join(out_dir, "input.txt")
    if not os.path.exists(raw):
        download(SHAKESPEARE_URL, raw)
    return split_text_file(raw, out_dir, args.val_fraction, args.max_train_bytes)


def prepare_text(out_dir, args):
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    return split_text_file(args.input, out_dir, args.val_fraction, args.max_train_bytes)


DATASETS = {
    "tinystories": prepare_tinystories,
    "shakespeare": prepare_shakespeare,
    "text": prepare_text,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare train.bin / val.bin for train.py --data-dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset", choices=DATASETS, help="dataset to prepare")
    p.add_argument("--input", help="text file to split (required for 'text')")
    p.add_argument(
        "--name", help="output subdirectory name (defaults to the dataset name)"
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "data"),
        help="parent directory for prepared datasets",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction held out for validation (shakespeare/text only)",
    )
    p.add_argument(
        "--max-train-bytes",
        type=lambda s: int(s.replace("_", "")),
        default=None,
        help="truncate the training split, e.g. 100_000_000 for a quick experiment",
    )
    args = p.parse_args()
    if args.dataset == "text" and not args.input:
        p.error("--input is required for the 'text' dataset")
    return args


def main():
    args = parse_args()
    name = args.name or (
        os.path.splitext(os.path.basename(args.input))[0]
        if args.dataset == "text"
        else args.dataset
    )
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preparing {args.dataset} into {out_dir}")

    train_bytes, val_bytes = DATASETS[args.dataset](out_dir, args)

    meta = {
        "dataset": args.dataset,
        "source": args.input,
        "tokenizer": "bytes",
        "vocab_size": 256,
        "dtype": "uint8",
        "train_tokens": train_bytes,
        "val_tokens": val_bytes,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"train.bin: {train_bytes:,} tokens ({fmt_bytes(train_bytes)})")
    print(f"val.bin:   {val_bytes:,} tokens ({fmt_bytes(val_bytes)})")
    print(f"Train with: python train.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
