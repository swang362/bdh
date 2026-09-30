# Copyright Pathway Technology, Inc.

"""Tokenizers shared by prepare_data.py, train.py and inference.py.

- ByteTokenizer: each UTF-8 byte is a token (vocab 256). The original setup,
  and the default for checkpoints/datasets that don't record a tokenizer.
- SentencePieceTokenizer: a BPE model trained by prepare_data.py. Special
  tokens such as <|endoftext|> are single tokens; byte fallback means any
  text can be encoded.

A tokenizer is stored inside checkpoints (see to_state / from_state), so a
checkpoint is self-contained for inference.
"""

import codecs
import json
import os

EOT = "<|endoftext|>"  # end-of-text marker, e.g. between TinyStories stories


class ByteTokenizer:
    type = "bytes"
    vocab_size = 256
    dtype = "uint8"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="backslashreplace")

    def token_id(self, piece):
        return None  # no multi-byte special tokens; EOT is 13 separate bytes

    def stream_decoder(self):
        return _ByteStreamDecoder()


class SentencePieceTokenizer:
    type = "sentencepiece"

    def __init__(self, model_proto):
        import sentencepiece as spm

        self.model_proto = bytes(model_proto)
        self.sp = spm.SentencePieceProcessor(model_proto=self.model_proto)
        self.vocab_size = self.sp.get_piece_size()
        self.dtype = "uint16" if self.vocab_size <= 2**16 else "uint32"

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(list(ids))

    def token_id(self, piece):
        # special tokens (user-defined symbols such as <|endoftext|>) are single pieces
        i = self.sp.piece_to_id(piece)
        return None if i == self.sp.unk_id() else i

    def stream_decoder(self):
        return _DiffStreamDecoder(self)


class _ByteStreamDecoder:
    """Turns byte tokens into text, holding back incomplete UTF-8 sequences."""

    def __init__(self):
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="backslashreplace")

    def feed(self, ids):
        return self._dec.decode(bytes(ids))

    def flush(self):
        return self._dec.decode(b"", final=True)


class _DiffStreamDecoder:
    """Decodes the whole sequence each time and returns only the new text.

    A token can end inside a multi-byte character (byte fallback), which decodes
    to U+FFFD until the rest arrives, so trailing U+FFFD is held back.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.emitted = 0

    def feed(self, ids):
        self.ids.extend(ids)
        text = self.tok.decode(self.ids)
        stable = len(text.rstrip("�"))
        new = text[self.emitted : stable]
        self.emitted = max(self.emitted, stable)
        return new

    def flush(self):
        text = self.tok.decode(self.ids)
        new = text[self.emitted :]
        self.emitted = len(text)
        return new


def to_state(tokenizer):
    """Tokenizer description to store in a checkpoint (torch.load weights_only-safe)."""
    if tokenizer.type == "bytes":
        return {"type": "bytes"}
    import torch

    proto = torch.frombuffer(bytearray(tokenizer.model_proto), dtype=torch.uint8)
    return {"type": tokenizer.type, "model": proto.clone()}


def from_state(state):
    """Inverse of to_state. Checkpoints without a tokenizer entry are byte-level."""
    if state is None or state["type"] == "bytes":
        return ByteTokenizer()
    if state["type"] == "sentencepiece":
        return SentencePieceTokenizer(state["model"].cpu().numpy().tobytes())
    raise ValueError(f"Unknown tokenizer type {state['type']!r}")


def same_tokenizer(a, b):
    return a.type == b.type and getattr(a, "model_proto", None) == getattr(
        b, "model_proto", None
    )


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return {"tokenizer": "bytes", "dtype": "uint8", "vocab_size": 256}
    with open(path) as f:
        return json.load(f)


def from_data_dir(data_dir):
    """Tokenizer used to prepare data_dir (see prepare_data.py)."""
    meta = load_meta(data_dir)
    if meta.get("tokenizer", "bytes") == "bytes":
        return ByteTokenizer()
    return SentencePieceTokenizer.from_file(
        os.path.join(data_dir, meta["tokenizer_model"])
    )
