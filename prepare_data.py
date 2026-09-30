# Copyright Pathway Technology, Inc.

"""Prepare a dataset as train.bin / val.bin for train.py --data-dir.

Two tokenizers are supported:
- bytes (default): each UTF-8 byte is a token (vocab 256), so the .bin files
  are simply the raw text, stored as uint8.
- sentencepiece: a BPE tokenizer is trained on the training split (or loaded
  with --tokenizer-model) and the text is encoded to uint16 token ids.
  Special tokens such as <|endoftext|> become single tokens.

Raw downloads are cached in <out-dir>/raw/<dataset>/, so preparing the same
dataset with another tokenizer doesn't download it again. Files are streamed
in chunks, so multi-GB datasets never need to fit in memory.

Wikipedia and FineWeb-Edu are published as parquet shards on Hugging Face;
--shards picks how many to download. Documents are joined into one text file,
separated by <|endoftext|> lines (the same format as TinyStories).

Examples:
    python prepare_data.py tinystories
    python prepare_data.py tinystories --tokenizer sentencepiece --vocab-size 4096
    python prepare_data.py tinystories --max-train-bytes 100_000_000
    python prepare_data.py shakespeare
    python prepare_data.py wikipedia --shards 2 --tokenizer sentencepiece --vocab-size 16384
    python prepare_data.py fineweb-edu --tokenizer sentencepiece --vocab-size 16384
    python prepare_data.py text --input my_corpus.txt --name my_corpus
"""

import argparse
import itertools
import json
import os

import numpy as np
import requests

from tokenizer import EOT, SentencePieceTokenizer

CHUNK_SIZE = 16 * 1024 * 1024
ENCODE_PIECE_SIZE = 1024 * 1024  # text per sentencepiece call, for multithreaded encoding
ROOT = os.path.dirname(os.path.abspath(__file__))

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"

# parquet datasets on Hugging Face: repo, folder with the shards, and whether to
# put the article title above the text
HF_DATASETS = {
    # English Wikipedia (2023-11-01 dump): 41 shards of about 330-420MB parquet each
    "wikipedia": {"repo": "wikimedia/wikipedia", "path": "20231101.en", "title": True},
    # FineWeb-Edu 10B-token sample: 14 shards of about 2.15GB parquet each
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "path": "sample/10BT", "title": False},
}
HF_VAL_FRACTION = 0.005  # these datasets are large, so hold out less by default


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def download(url, dst):
    """Stream url to dst unless it already exists."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    written = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                pct = f" ({100 * written / total:.0f}%)" if total else ""
                print(f"\r  {os.path.basename(dst)}: {fmt_bytes(written)}{pct}", end="")
    print()
    os.replace(tmp, dst)


# A source is (path, start, length): a byte range of a text file.


def whole_file(path):
    return (path, 0, os.path.getsize(path))


def align_to_line(path, pos):
    """Move pos forward to just after the next newline, so ranges split whole lines."""
    size = os.path.getsize(path)
    if pos <= 0 or pos >= size:
        return max(0, min(pos, size))
    with open(path, "rb") as f:
        f.seek(pos)
        while True:
            block = f.read(1024 * 1024)
            if not block:
                return size
            i = block.find(b"\n")
            if i >= 0:
                return pos + i + 1
            pos += len(block)


def split_file(path, val_fraction):
    """Split one text file at a line boundary: first part train, last part val."""
    size = os.path.getsize(path)
    cut = align_to_line(path, int(size * (1 - val_fraction)))
    return (path, 0, cut), (path, cut, size - cut)


def truncate(src, max_bytes):
    path, start, length = src
    if max_bytes is None or length <= max_bytes:
        return src
    return (path, start, align_to_line(path, start + max_bytes) - start)


def read_chunks(src, text=False):
    """Yield the bytes (or text, split at line boundaries) of a source range."""
    path, start, length = src
    remaining = length
    carry = b""
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            if not text:
                yield chunk
                continue
            chunk = carry + chunk
            # keep a trailing partial line for the next chunk so no line (or
            # multi-byte character) is split between two encode calls
            cut = chunk.rfind(b"\n") + 1 if remaining > 0 else len(chunk)
            if cut == 0:
                carry = chunk
                continue
            carry = chunk[cut:]
            yield chunk[:cut].decode("utf-8", errors="replace")
    if text and carry:
        yield carry.decode("utf-8", errors="replace")


def write_bytes(src, dst):
    """Byte tokenizer: the .bin file is the raw text. Returns tokens written."""
    written = 0
    with open(dst, "wb") as f:
        for chunk in read_chunks(src):
            f.write(chunk)
            written += len(chunk)
    return written


def split_pieces(text, size):
    """Split text into pieces of about size characters at line boundaries."""
    pieces, start = [], 0
    while start < len(text):
        end = text.find("\n", start + size)
        end = len(text) if end < 0 else end + 1
        pieces.append(text[start:end])
        start = end
    return pieces


def write_tokens(tok, src, dst):
    """Encode a source range with sentencepiece into dst. Returns tokens written."""
    written = 0
    done = 0
    threads = os.cpu_count() or 1
    with open(dst, "wb") as f:
        for text in read_chunks(src, text=True):
            pieces = split_pieces(text, ENCODE_PIECE_SIZE)
            try:
                ids = tok.sp.encode(pieces, num_threads=threads)
            except TypeError:  # older sentencepiece without num_threads
                ids = tok.sp.encode(pieces)
            n = sum(len(x) for x in ids)
            np.fromiter(itertools.chain.from_iterable(ids), dtype=tok.dtype, count=n).tofile(f)
            written += n
            done += len(text.encode("utf-8"))
            print(
                f"\r  {os.path.basename(dst)}: {written:,} tokens "
                f"({100 * done / max(1, src[2]):.0f}%)",
                end="",
            )
    print()
    return written


def train_sentencepiece(src, out_dir, args):
    import sentencepiece as spm

    path, start, length = src
    input_path = path
    if start != 0 or length != os.path.getsize(path):
        # train only on the training range (never on validation text)
        input_path = os.path.join(out_dir, "tokenizer_train.txt")
        write_bytes(src, input_path)
    prefix = os.path.join(out_dir, "tokenizer")
    print(f"Training sentencepiece BPE tokenizer, vocab {args.vocab_size}")
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        user_defined_symbols=args.special_tokens,  # always kept as single tokens
        byte_fallback=True,  # unknown characters become byte tokens, never <unk>
        character_coverage=1.0,
        # keep the text exactly as is, so decode(encode(text)) == text
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        bos_id=-1,
        eos_id=-1,
        input_sentence_size=args.tokenizer_sample_lines,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=os.cpu_count() or 1,
    )
    if input_path != path:
        os.remove(input_path)
    return prefix + ".model"


def sources_tinystories(args):
    # TinyStories ships its own train/valid split; stories are separated by <|endoftext|>
    raw = os.path.join(args.out_dir, "raw", "tinystories")
    train, val = os.path.join(raw, "train.txt"), os.path.join(raw, "valid.txt")
    download(TINYSTORIES_URL.format(split="train"), train)
    download(TINYSTORIES_URL.format(split="valid"), val)
    return whole_file(train), whole_file(val)


def sources_shakespeare(args):
    raw = os.path.join(args.out_dir, "raw", "shakespeare", "input.txt")
    download(SHAKESPEARE_URL, raw)
    return split_file(raw, args.val_fraction)


def sources_text(args):
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    return split_file(args.input, args.val_fraction)


def list_hf_shards(repo, path):
    """Parquet shard paths of a Hugging Face dataset folder, in order."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return sorted(
        f["path"] for f in r.json() if f["type"] == "file" and f["path"].endswith(".parquet")
    )


def parquet_to_text(parquet_path, out, with_title):
    """Append each document of a parquet shard to out, followed by an <|endoftext|> line."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("Reading parquet needs pyarrow: pip install pyarrow")
    columns = ["title", "text"] if with_title else ["text"]
    docs = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024, columns=columns):
        texts = batch.column("text").to_pylist()
        titles = batch.column("title").to_pylist() if with_title else [None] * len(texts)
        for title, text in zip(titles, texts):
            text = (text or "").strip()
            if not text:
                continue
            doc = f"{title}\n\n{text}" if title else text
            out.write(f"{doc}\n{EOT}\n".encode("utf-8"))
            docs += 1
    return docs


def sources_hf(args):
    spec = HF_DATASETS[args.dataset]
    raw = os.path.join(args.out_dir, "raw", args.dataset)
    shards = list_hf_shards(spec["repo"], spec["path"])
    if args.shards < 1 or args.shards > len(shards):
        raise SystemExit(f"--shards must be between 1 and {len(shards)} for {args.dataset}")
    shards = shards[: args.shards]
    # the joined text is cached per shard count; parquet shards are kept for reuse
    text_path = os.path.join(raw, f"text_{len(shards)}shards.txt")
    if not os.path.exists(text_path):
        os.makedirs(raw, exist_ok=True)
        tmp = text_path + ".part"
        with open(tmp, "wb") as out:
            for i, shard in enumerate(shards):
                local = os.path.join(raw, os.path.basename(shard))
                download(f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{shard}", local)
                print(f"  converting shard {i + 1}/{len(shards)} to text...", end="", flush=True)
                docs = parquet_to_text(local, out, spec["title"])
                print(f" {docs:,} documents, {fmt_bytes(out.tell())} so far")
        os.replace(tmp, text_path)
    return split_file(text_path, args.val_fraction)


DATASETS = {
    "tinystories": sources_tinystories,
    "shakespeare": sources_shakespeare,
    "wikipedia": sources_hf,
    "fineweb-edu": sources_hf,
    "text": sources_text,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare train.bin / val.bin for train.py --data-dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset", choices=DATASETS, help="dataset to prepare")
    p.add_argument("--input", help="text file to split (required for 'text')")
    p.add_argument(
        "--name",
        help="output subdirectory name (default: dataset name, plus _sp<vocab> for sentencepiece)",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "data"),
        help="parent directory for prepared datasets (raw downloads go to <out-dir>/raw)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=f"fraction held out for validation (not tinystories, which has its own split); "
        f"default 0.1, or {HF_VAL_FRACTION} for wikipedia/fineweb-edu",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=1,
        help="parquet shards to download (wikipedia: up to 41, about 330-420MB each; "
        "fineweb-edu: up to 14, about 2.15GB each)",
    )
    p.add_argument(
        "--max-train-bytes",
        type=lambda s: int(s.replace("_", "")),
        default=None,
        help="truncate the training split, e.g. 100_000_000 for a quick experiment",
    )
    g = p.add_argument_group("tokenizer")
    g.add_argument("--tokenizer", choices=["bytes", "sentencepiece"], default="bytes")
    g.add_argument("--vocab-size", type=int, default=4096, help="sentencepiece vocab size")
    g.add_argument(
        "--special-tokens",
        nargs="*",
        default=[EOT],
        help="sentencepiece tokens that are never split",
    )
    g.add_argument(
        "--tokenizer-model",
        help="reuse an existing sentencepiece .model instead of training one",
    )
    g.add_argument(
        "--tokenizer-sample-lines",
        type=int,
        default=2_000_000,
        help="lines sampled from the training split to train the tokenizer",
    )
    args = p.parse_args()
    if args.dataset == "text" and not args.input:
        p.error("--input is required for the 'text' dataset")
    if args.val_fraction is None:
        args.val_fraction = HF_VAL_FRACTION if args.dataset in HF_DATASETS else 0.1
    return args


def main():
    args = parse_args()
    name = args.name or (
        os.path.splitext(os.path.basename(args.input))[0]
        if args.dataset == "text"
        else args.dataset
    )
    if not args.name and args.dataset in HF_DATASETS and args.shards != 1:
        name += f"_{args.shards}shards"
    if not args.name and args.tokenizer == "sentencepiece":
        name += f"_sp{args.vocab_size}" if not args.tokenizer_model else "_sp"
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preparing {args.dataset} into {out_dir} ({args.tokenizer} tokenizer)")

    train_src, val_src = DATASETS[args.dataset](args)
    train_src = truncate(train_src, args.max_train_bytes)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    meta = {"dataset": args.dataset, "source": args.input}
    if args.tokenizer == "bytes":
        train_tokens = write_bytes(train_src, train_path)
        val_tokens = write_bytes(val_src, val_path)
        meta.update(tokenizer="bytes", vocab_size=256, dtype="uint8")
    else:
        model_path = os.path.join(out_dir, "tokenizer.model")
        if args.tokenizer_model:
            with open(args.tokenizer_model, "rb") as fin, open(model_path, "wb") as fout:
                fout.write(fin.read())
        else:
            train_sentencepiece(train_src, out_dir, args)
        tok = SentencePieceTokenizer.from_file(model_path)
        train_tokens = write_tokens(tok, train_src, train_path)
        val_tokens = write_tokens(tok, val_src, val_path)
        meta.update(
            tokenizer="sentencepiece",
            tokenizer_model="tokenizer.model",
            vocab_size=tok.vocab_size,
            dtype=tok.dtype,
            special_tokens=[t for t in args.special_tokens if tok.token_id(t) is not None],
        )

    train_bytes, val_bytes = train_src[2], val_src[2]
    meta.update(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_bytes=train_bytes,
        val_bytes=val_bytes,
        # for comparing losses across tokenizers: bits/byte = loss / ln(2) / bytes_per_token
        bytes_per_token=train_bytes / max(1, train_tokens),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"train.bin: {train_tokens:,} tokens from {fmt_bytes(train_bytes)}")
    print(f"val.bin:   {val_tokens:,} tokens from {fmt_bytes(val_bytes)}")
    print(f"{meta['bytes_per_token']:.2f} bytes/token, vocab {meta['vocab_size']}")
    print(f"Train with: python train.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
