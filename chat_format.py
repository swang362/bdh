# Copyright Pathway Technology, Inc.

"""Chat format shared by finetune.py and chat.py.

Conversations are rendered as plain text with role headers, and every
assistant turn ends with <|endoftext|>:

    ### System:
    You are a helpful assistant.

    ### User:
    What is the capital of France?

    ### Assistant:
    The capital of France is Paris.<|endoftext|>

The headers are ordinary text, so any base checkpoint and tokenizer works
(bytes or sentencepiece) without adding tokens. During fine-tuning only the
assistant content (and its <|endoftext|>) is trained on; everything else is
context.
"""

from tokenizer import EOT

FORMAT_VERSION = "bdh-chat-v1"
HEADERS = {"system": "### System:\n", "user": "### User:\n", "assistant": "### Assistant:\n"}
TURN_SEP = "\n\n"
# chat.py stops generating when the model starts a new turn by itself
STOP_STRINGS = [EOT, "\n### User:", "\n### System:"]


def render(messages, add_generation_prompt=False):
    """Render messages as a list of (text, is_trained) segments.

    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...].
    With add_generation_prompt, an assistant header is appended so the model
    continues with the answer (used by chat.py).
    """
    segments = []
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role not in HEADERS:
            raise ValueError(f"Unknown role {role!r}")
        if role == "assistant":
            segments.append((HEADERS[role], False))
            segments.append((content + EOT, True))
            segments.append((TURN_SEP, False))
        else:
            segments.append((HEADERS[role] + content + TURN_SEP, False))
    if add_generation_prompt:
        segments.append((HEADERS["assistant"], False))
    return segments


def encode(tok, messages, add_generation_prompt=False):
    """Token ids and a same-length mask: True where the token is trained on.

    Segments are encoded separately, so the prompt part of a training example
    tokenizes exactly like the prompt chat.py builds at inference time.
    """
    ids, mask = [], []
    for text, trained in render(messages, add_generation_prompt):
        seg = tok.encode(text)
        ids += seg
        mask += [trained] * len(seg)
    return ids, mask
