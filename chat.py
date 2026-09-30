# Copyright Pathway Technology, Inc.

"""Chat / question answering with a fine-tuned BDH checkpoint (see finetune.py).

Examples:
    python chat.py                                   # interactive, checkpoints/chat/best.pt
    python chat.py "What is the capital of France?"  # one question, then exit
    python chat.py --checkpoint checkpoints/wiki_chat/best.pt --system "Answer briefly."
    python chat.py --recurrent --cuda-graph --state session.pt   # conversation continues across runs

Interactive commands: /reset clears the conversation, /save FILE and /load FILE store
and restore it (--recurrent), /exit (or Ctrl+D) quits.
"""

import argparse
import os
import sys
from contextlib import nullcontext

import bdh
import torch

import chat_format
import tokenizer as tokenizers
from inference import StopAtText, token_stream
from recurrent import RecurrentBDH

DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "chat", "best.pt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Chat with a fine-tuned BDH checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", nargs="?", help="ask one question and exit (default: interactive)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    p.add_argument("--system", default=None, help="system prompt (default: the one used in fine-tuning, if any)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--context-size", type=int, default=None, help="context window (default: the checkpoint's block size)")
    p.add_argument("--no-history", action="store_true", help="answer each question independently")
    p.add_argument(
        "--recurrent",
        action="store_true",
        help="generate recurrently with a fixed-size state: faster, especially in long conversations (see docs/recurrent.md)",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="with --recurrent on CUDA: replay each token step as a recorded CUDA graph (faster); recorded once per session",
    )
    p.add_argument(
        "--state",
        metavar="FILE",
        help="with --recurrent: continue the conversation saved in FILE if it exists, and save it there on exit "
        "(also the default file for /save and /load)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    return p.parse_args()


def load_model(path, device):
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path} (run finetune.py first)")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = bdh.BDH(bdh.BDHConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tok = tokenizers.from_state(checkpoint.get("tokenizer"))
    info = {
        "step": checkpoint["step"],
        "block_size": checkpoint.get("block_size", 512),
        "chat": checkpoint.get("chat"),
    }
    return model, tok, info


def generate_reply(
    model, tok, messages, context_size, max_new_tokens=256, temperature=0.7, top_k=20, on_text=None,
    recurrent=False, cuda_graph=False, state=None, resume=False,
):
    """Generate the assistant's reply to messages; on_text receives text as it streams.

    In recurrent mode, pass the same RecurrentBDH as state every turn to reuse its
    buffers and recorded CUDA graph; it is reset and re-read from messages. With
    resume=True, messages are ignored and the reply continues from what the state
    already holds (used by RecurrentSession).
    """
    ids = [] if resume else chat_format.encode(tok, messages, add_generation_prompt=True)[0]
    device = next(model.parameters()).device
    decoder = tok.stream_decoder()
    decoder.feed(ids)  # the prompt itself is not printed
    stop = StopAtText(chat_format.STOP_STRINGS)
    pieces = []

    def emit(text):
        if text:
            pieces.append(text)
            if on_text:
                on_text(text)

    for token in token_stream(
        model, ids, device, recurrent, max_new_tokens, temperature, top_k, context_size or None,
        cuda_graph=cuda_graph, state=state, resume=resume,
    ):
        emit(stop.feed(decoder.feed([token])))
        if stop.stopped:
            break
    if not stop.stopped:
        emit(stop.feed(decoder.flush()) + stop.flush())
    return "".join(pieces).strip()


class RecurrentSession:
    """A conversation kept in one recurrent state: each turn reads only the new tokens.

    The state always holds exactly the conversation as chat_format renders it, so a
    session gives the same replies as re-reading the whole conversation every turn
    (within the context window), just without the re-reading. It can be saved to a
    file and loaded later, including the message history.
    """

    def __init__(self, state, tok, system):
        self.state = state
        self.tok = tok
        self.system = system
        self.history = []

    def reset(self):
        self.state.reset()
        self.history = []

    def ask(self, question, generate):
        """Add a question, generate the reply with generate() (resume mode), store both."""
        user = {"role": "user", "content": question}
        new = ([{"role": "system", "content": self.system}] if self.system and self.state.pos == 0 else []) + [user]
        new_ids, _ = chat_format.encode(self.tok, new, add_generation_prompt=True)
        if self.state.pos == 0:
            self.state.prefill(new_ids)
        else:
            self.state.feed(new_ids)
        snapshot = self.state.snapshot()
        n_before = len(self.state.ids)
        reply = generate()
        # Store the reply exactly as the rendered conversation would tokenize it (content,
        # <|endoftext|>, turn separator). The generated tokens usually are a prefix of that;
        # if not (different tokenization, or the start of a stop string was read), go back
        # to before the reply and read the canonical version instead.
        canonical = self.tok.encode(reply.strip() + tokenizers.EOT) + self.tok.encode(chat_format.TURN_SEP)
        fed = self.state.ids[n_before:]
        if canonical[: len(fed)] == fed:
            self.state.feed(canonical[len(fed) :])
        else:
            self.state.restore(snapshot)
            self.state.feed(canonical)
        self.history += [user, {"role": "assistant", "content": reply}]
        return reply

    def save(self, path):
        self.state.save(path, extra={"chat_format": chat_format.FORMAT_VERSION, "system": self.system, "messages": self.history})

    def load(self, path):
        extra = self.state.load(path)
        self.history = list(extra.get("messages", []))
        self.system = extra.get("system", self.system)


def main():
    args = parse_args()
    if args.cuda_graph and not args.recurrent:
        raise SystemExit("--cuda-graph requires --recurrent")
    if args.state and (not args.recurrent or args.no_history):
        raise SystemExit("--state requires --recurrent (and conversation history, so not --no-history)")
    sys.stdout.reconfigure(errors="backslashreplace")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_cuda else nullcontext()

    model, tok, info = load_model(args.checkpoint, device)
    context_size = args.context_size if args.context_size is not None else info["block_size"]
    system = args.system if args.system is not None else (info["chat"] or {}).get("system")
    print(
        f"Loaded {args.checkpoint} (step {info['step']}) on {device}, context {context_size or 'unlimited'}"
        + (", recurrent" if args.recurrent else "")
        + (" with CUDA graph" if args.recurrent and args.cuda_graph else ""),
        file=sys.stderr,
    )
    if not info["chat"]:
        print(
            "Warning: this checkpoint was not fine-tuned for chat (run finetune.py); "
            "replies will be plain text continuation.",
            file=sys.stderr,
        )
    # one recurrent state for the whole session: its buffers and CUDA graph are reused every turn
    state = RecurrentBDH(model, window=context_size or None, cuda_graph=args.cuda_graph) if args.recurrent else None
    # with history, the conversation lives in the state and only new tokens are read each turn
    session = RecurrentSession(state, tok, system) if state is not None and not args.no_history else None
    write = lambda t: (sys.stdout.write(t), sys.stdout.flush())

    def ask(history, question):
        if session is not None:
            with ctx:
                reply = session.ask(
                    question,
                    lambda: generate_reply(
                        model, tok, None, context_size, args.max_new_tokens, args.temperature, args.top_k,
                        on_text=write, recurrent=True, state=state, resume=True,
                    ),
                )
            print()
            return reply
        messages = ([{"role": "system", "content": system}] if system else []) + history
        messages.append({"role": "user", "content": question})
        with ctx:
            reply = generate_reply(
                model, tok, messages, context_size, args.max_new_tokens, args.temperature, args.top_k,
                on_text=write, recurrent=args.recurrent, state=state,
            )
        print()
        return reply

    def load_session(path):
        try:
            session.load(path)
        except (OSError, ValueError) as e:
            print(f"(can't load {path}: {e})", file=sys.stderr)
            return False
        turns = len(session.history) // 2
        print(f"(loaded {path}: {turns} turns, {state.pos:,} tokens read)", file=sys.stderr)
        return True

    def save_session(path):
        session.save(path)
        print(f"(saved {path}: {len(session.history) // 2} turns, {state.pos:,} tokens)", file=sys.stderr)

    if session is not None and args.state and os.path.exists(args.state):
        load_session(args.state)

    if args.question:
        ask([], args.question)
        if session is not None and args.state:
            save_session(args.state)
        return

    print(
        "Chat started. /reset clears the conversation, "
        + ("/save FILE and /load FILE store and restore it, " if session is not None else "")
        + "/exit quits.",
        file=sys.stderr,
    )
    history = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/reset":
            history = []
            if session is not None:
                session.reset()
            print("(conversation cleared)", file=sys.stderr)
            continue
        if question.split()[0] in ("/save", "/load"):
            command, _, path = question.partition(" ")
            path = path.strip() or args.state
            if session is None:
                print("(/save and /load need --recurrent and conversation history)", file=sys.stderr)
            elif not path:
                print(f"(usage: {command} FILE)", file=sys.stderr)
            elif command == "/save":
                save_session(path)
            else:
                load_session(path)
            continue
        print("Assistant: ", end="", flush=True)
        try:
            reply = ask([] if args.no_history else history, question)
        except KeyboardInterrupt:
            # the recurrent state may hold a half-read turn: start the conversation over
            print("\n(interrupted; conversation cleared)" if session is not None else "\n(interrupted)", file=sys.stderr)
            if session is not None:
                session.reset()
            continue
        if not args.no_history and session is None:
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]

    if session is not None and args.state:
        save_session(args.state)


if __name__ == "__main__":
    main()
