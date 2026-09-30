# Copyright Pathway Technology, Inc.

import argparse
import codecs
import os
import sys
import time
from contextlib import nullcontext

import bdh
import torch
import torch.nn.functional as F

DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "latest.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a BDH checkpoint")
    parser.add_argument("prompt", help="text to continue")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=200)
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
        "--cpu", action="store_true", help="run on CPU even if a GPU is available"
    )
    parser.add_argument(
        "--resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print CPU/GPU resource usage after generation",
    )
    return parser.parse_args()


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
def generate_stream(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    # same sampling as BDH.generate, but yields each new token as it is produced
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        yield idx_next.item()


def main():
    args = parse_args()
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
    model = bdh.BDH(bdh.BDHConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(
        f"Loaded {args.checkpoint} (step {checkpoint['step']}) on {device}",
        file=sys.stderr,
    )

    # byte-level "tokenizer": each UTF-8 byte is one token
    prompt = torch.tensor(
        bytearray(args.prompt, "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
        # measure peak GPU memory of generation only, not of checkpoint loading
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    cpu_start = time.process_time()
    first_token_time = None
    with ctx:
        if args.stream:
            # a multi-byte UTF-8 character spans several tokens, so decode incrementally
            decoder = codecs.getincrementaldecoder("utf-8")(errors="backslashreplace")
            sys.stdout.write(args.prompt)
            sys.stdout.flush()
            for token in generate_stream(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                sys.stdout.write(decoder.decode(bytes([token])))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True) + "\n")
        else:
            out = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            text = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode(
                errors="backslashreplace"
            )
            print(text)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cpu_time = time.process_time() - cpu_start

    # stats go to stderr so stdout holds only the generated text
    stats = (
        f"\nGenerated {args.max_new_tokens} tokens in {elapsed:.2f}s "
        f"({args.max_new_tokens / elapsed:.1f} tok/s)"
    )
    if first_token_time is not None:
        stats += f", first token after {first_token_time * 1000:.0f}ms"
    print(stats, file=sys.stderr)
    if args.resources:
        print_resources(model, device, elapsed, cpu_time)


if __name__ == "__main__":
    main()
