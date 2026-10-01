# Copyright Pathway Technology, Inc.

import argparse
import os
import sys
import time
from contextlib import nullcontext

import models
import tokenizer as tokenizers
import torch
from recurrent import RecurrentBDH, generate_stream_recurrent, sample_next

DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "latest.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a BDH checkpoint")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help=f"text to continue; may contain special tokens such as {tokenizers.EOT}. "
        "May be empty with --load-state, to continue straight from the saved state",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="generate recurrently with a fixed-size state instead of re-reading the context for every token (see docs/recurrent.md)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="with --recurrent on CUDA: record each token step as a CUDA graph and replay it with one launch (faster; see docs/recurrent.md)",
    )
    parser.add_argument(
        "--load-state",
        metavar="FILE",
        help="with --recurrent: start from a saved state; the prompt is appended after it",
    )
    parser.add_argument(
        "--save-state",
        metavar="FILE",
        help="with --recurrent: save the state after generating (including the generated text), to continue later",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="max tokens fed to the model per step (default: the training block size "
        "saved in the checkpoint; 0 means unlimited)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print tokens as they are generated instead of all at once",
    )
    parser.add_argument(
        "--stop-at-eot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"stop when the model generates {tokenizers.EOT}",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run on CPU even if a GPU is available"
    )
    parser.add_argument(
        "--resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print CPU/GPU resource usage after generation",
    )
    return parser.parse_args()


class StopAtText:
    """Passes text through until a stop string appears, then drops it and the rest.

    stops is one string, a list of strings, or None. A suffix that could be the
    start of a stop string is held back, so a partially generated <|endoftext|>
    is never printed.
    """

    def __init__(self, stops):
        if isinstance(stops, str):
            stops = [stops]
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.stopped = False

    def feed(self, text):
        if not self.stops or self.stopped:
            return "" if self.stopped else text
        self.buf += text
        hits = [i for i in (self.buf.find(s) for s in self.stops) if i >= 0]
        if hits:
            self.stopped = True
            out, self.buf = self.buf[: min(hits)], ""
            return out
        keep = 0
        for stop in self.stops:
            for k in range(min(len(stop) - 1, len(self.buf)), keep, -1):
                if stop.startswith(self.buf[-k:]):
                    keep = k
                    break
        out = self.buf[: len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep :]
        return out

    def flush(self):
        out, self.buf = self.buf, ""
        return out


def fmt_mb(n):
    return f"{n / 2**20:,.1f}MB"


def peak_rss():
    """Peak resident memory of this process in bytes, or None if unavailable."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        # peak_wset only exists on Windows; elsewhere fall back to current rss
        return getattr(info, "peak_wset", info.rss)
    except ImportError:
        pass
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss if sys.platform == "darwin" else maxrss * 1024  # KB on Linux
    except ImportError:
        return None  # Windows without psutil


def print_resources(model, device, elapsed, cpu_time):
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    lines = [
        f"Model: {n_params:,} params ({fmt_mb(param_bytes)} weights)",
        # CPU time can exceed wall time when torch uses several threads
        f"CPU: {cpu_time:.2f}s CPU time ({100 * cpu_time / elapsed:.0f}% of one core), "
        f"{torch.get_num_threads()} torch threads",
    ]
    rss = peak_rss()
    lines.append(
        f"RAM: peak {fmt_mb(rss)}" if rss is not None
        else "RAM: unavailable (pip install psutil)"
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        lines.append(
            f"GPU: {props.name}, peak allocated {fmt_mb(torch.cuda.max_memory_allocated(device))}, "
            f"reserved {fmt_mb(torch.cuda.memory_reserved(device))} "
            f"of {fmt_mb(props.total_memory)}"
        )
    print("\n".join(lines), file=sys.stderr)


@torch.no_grad()
def generate_stream(model, idx, max_new_tokens, temperature=1.0, top_k=None, context_size=None):
    # same sampling as BDH.generate, but yields each new token as it is produced
    for _ in range(max_new_tokens):
        # the attention sums over all previous tokens without normalization, so
        # sequences longer than the training block size go out of distribution;
        # feed only the most recent context_size tokens (a sliding window)
        idx_cond = idx if context_size is None else idx[:, -context_size:]
        logits, _ = model(idx_cond)
        idx_next = sample_next(logits[:, -1, :], temperature, top_k)
        idx = torch.cat((idx, idx_next), dim=1)
        yield idx_next.item()


def token_stream(
    model, prompt_ids, device, recurrent, max_new_tokens, temperature, top_k, context_size, cuda_graph=False,
    state=None, resume=False,
):
    """Generated token ids, by re-reading the window each step or recurrently (see recurrent.py).

    cuda_graph, state and resume only apply to recurrent mode: state is a RecurrentBDH to
    reuse (keeping its recorded CUDA graph), e.g. across chat turns; with resume, the
    prompt is fed after what the state already holds instead of resetting it.
    """
    if recurrent:
        return generate_stream_recurrent(
            model, prompt_ids, max_new_tokens, temperature=temperature, top_k=top_k,
            context_size=context_size, state=state, cuda_graph=cuda_graph, resume=resume,
        )
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    return generate_stream(
        model, idx, max_new_tokens, temperature=temperature, top_k=top_k, context_size=context_size
    )


def main():
    args = parse_args()
    if args.cuda_graph and not args.recurrent:
        raise SystemExit("--cuda-graph requires --recurrent")
    if (args.load_state or args.save_state) and not args.recurrent:
        raise SystemExit("--load-state and --save-state require --recurrent")
    # a redirected stdout on Windows may be cp1252; escape unencodable chars instead of crashing
    sys.stdout.reconfigure(errors="backslashreplace")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint} (run train.py first)")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if args.recurrent and models.arch_of(checkpoint) != "bdh":
        raise SystemExit("--recurrent needs a BDH checkpoint; this one is " + models.arch_of(checkpoint))
    model = models.from_checkpoint(checkpoint).to(device)
    model.eval()
    # checkpoints from before tokenizer support have no entry and are byte-level
    tok = tokenizers.from_state(checkpoint.get("tokenizer"))
    context_size = args.context_size
    if context_size is None:
        # checkpoints from before block_size was saved: assume train.py's default
        context_size = checkpoint.get("block_size", 512)
        if "block_size" not in checkpoint:
            print(
                "Checkpoint has no block_size; assuming 512 "
                "(pass --context-size if you trained with a different --block-size)",
                file=sys.stderr,
            )
    print(
        f"Loaded {args.checkpoint} (step {checkpoint['step']}) on {device}, "
        f"{tok.type} tokenizer (vocab {tok.vocab_size}), "
        f"context {context_size or 'unlimited'}"
        + (", recurrent" if args.recurrent else "")
        + (" with CUDA graph" if args.recurrent and args.cuda_graph else ""),
        file=sys.stderr,
    )

    state = None
    if args.recurrent:
        state = RecurrentBDH(model, window=context_size or None, cuda_graph=args.cuda_graph)
        if args.load_state:
            try:
                state.load(args.load_state)
            except (OSError, ValueError) as e:
                raise SystemExit(f"Can't load state: {e}")
            print(f"Loaded state {args.load_state} ({state.pos:,} tokens read so far)", file=sys.stderr)

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids and not args.load_state:
        raise SystemExit("Prompt is empty")
    # tokens in the state once the prompt is read (recurrent mode), to check below
    tokens_before = (state.pos if args.load_state else 0) + len(prompt_ids)

    # a character can span several tokens (multi-byte UTF-8, byte fallback),
    # so text is decoded incrementally; the stop filter only sees generated text
    decoder = tok.stream_decoder()
    prompt_text = decoder.feed(prompt_ids)
    stop = StopAtText(tokenizers.EOT if args.stop_at_eot else None)
    pieces = []

    def emit(text):
        if args.stream:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            pieces.append(text)

    if device.type == "cuda":
        torch.cuda.synchronize()
        # measure peak GPU memory of generation only, not of checkpoint loading
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    cpu_start = time.process_time()
    first_token_time = None
    n_generated = 0
    last_token = None
    emit(prompt_text)
    with ctx:
        for token in token_stream(
            model,
            prompt_ids,
            device,
            args.recurrent,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            context_size or None,
            cuda_graph=args.cuda_graph,
            state=state,
            resume=bool(args.load_state),
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            n_generated += 1
            last_token = token
            emit(stop.feed(decoder.feed([token])))
            if stop.stopped:
                break
    emit(stop.feed(decoder.flush()) + stop.flush() + "\n")
    if not args.stream:
        print("".join(pieces), end="")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cpu_time = time.process_time() - cpu_start

    if args.save_state:
        # the last generated token is only fed when the next one is requested; feed it
        # now, so a continued run picks up after everything that was generated
        with ctx:
            if last_token is not None and len(state.ids) < tokens_before + n_generated:
                state.step(last_token)
        state.save(args.save_state)
        print(
            f"Saved state {args.save_state} ({state.pos:,} tokens read, "
            f"{state.state_bytes() / 2**20:,.0f}MB)",
            file=sys.stderr,
        )

    # stats go to stderr so stdout holds only the generated text
    stats = (
        f"\nGenerated {n_generated} tokens in {elapsed:.2f}s "
        f"({n_generated / elapsed:.1f} tok/s)"
    )
    if first_token_time is not None and args.stream:
        stats += f", first token after {first_token_time * 1000:.0f}ms"
    if stop.stopped:
        stats += f", stopped at {tokenizers.EOT}"
    print(stats, file=sys.stderr)
    if args.resources:
        print_resources(model, device, elapsed, cpu_time)


if __name__ == "__main__":
    main()
