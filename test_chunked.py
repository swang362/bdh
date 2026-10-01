# Copyright Pathway Technology, Inc.

"""Checks that chunked attention (bdh.ChunkedAttention, train.py --attn-chunk)
matches full attention, in the forward and the backward pass:

    python test_chunked.py

With a CUDA GPU it also compares them under bf16 autocast, as in training.
--bench times a training step (forward + backward) and peak memory with full
and chunked attention for a few block sizes:

    python test_chunked.py --bench --n-embd 512 --batch-size 4
"""

import argparse
import time

import torch

import bdh
from bdh import ChunkedAttention

TOLERANCE = 1e-6  # float64 checks, max |difference| relative to the largest value


def rel_diff(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def check(name, diff, tolerance=TOLERANCE):
    ok = diff < tolerance
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: max relative difference {diff:.2e}")
    return ok


def full_attention(Q, V):
    return (Q @ Q.mT).tril(diagonal=-1) @ V


def check_function(B, nh, T, N, D, chunk, device):
    """ChunkedAttention vs the full formula on random inputs, float64."""
    gen = torch.Generator().manual_seed(T * 31 + chunk)
    Q = torch.randn(B, nh, T, N, generator=gen, dtype=torch.float64).to(device).requires_grad_()
    V = torch.randn(B, 1, T, D, generator=gen, dtype=torch.float64).to(device).requires_grad_()
    dO = torch.randn(B, nh, T, D, generator=gen, dtype=torch.float64).to(device)
    ref = full_attention(Q, V)
    ref_dQ, ref_dV = torch.autograd.grad(ref, (Q, V), dO)
    out = ChunkedAttention.apply(Q, V, chunk)
    dQ, dV = torch.autograd.grad(out, (Q, V), dO)
    name = f"T={T}, chunk={chunk}"
    ok = check(f"{name}: output", rel_diff(out, ref))
    ok &= check(f"{name}: grad Q", rel_diff(dQ, ref_dQ))
    ok &= check(f"{name}: grad V", rel_diff(dV, ref_dV))
    return ok


def model_grads(model, idx, targets, chunk, ctx=None):
    model.attn.chunk_size = chunk
    model.zero_grad(set_to_none=True)
    with ctx or torch.autocast(idx.device.type, enabled=False):
        _, loss = model(idx, targets)
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def check_model(model, T, chunk, device, autocast=False):
    """A whole BDH training step: loss and every parameter gradient."""
    gen = torch.Generator().manual_seed(1)
    idx = torch.randint(0, model.config.vocab_size, (2, T + 1), generator=gen).to(device)
    x, y = idx[:, :-1], idx[:, 1:]
    ctx = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else None
    ref_loss, ref = model_grads(model, x, y, None, ctx)
    loss, got = model_grads(model, x, y, chunk, ctx)
    tol = 3e-2 if autocast else 1e-4
    label = f"T={T}, chunk={chunk}" + (", bf16 autocast" if autocast else ", float32")
    ok = check(f"{label}: loss", (abs(loss - ref_loss) / ref_loss).item(), tol)
    worst = max(rel_diff(got[n], ref[n]) for n in ref)
    ok &= check(f"{label}: parameter gradients (worst)", worst, tol)
    model.attn.chunk_size = None
    return ok


def bench(args):
    device = torch.device("cuda")
    cfg = bdh.BDHConfig(n_embd=args.n_embd, n_head=4, dropout=0.0, vocab_size=args.vocab)
    model = bdh.BDH(cfg).to(device)
    print(f"\nTraining step timing: n_embd {args.n_embd}, batch {args.batch_size}, bf16 autocast")
    print(f"{'block':>6} {'attention':>12} {'ms/step':>9} {'tok/s':>9} {'peak GB':>8}")
    for T in args.bench_blocks:
        for chunk in [None] + args.bench_chunks:
            if chunk and chunk >= T:
                continue
            model.attn.chunk_size = chunk
            idx = torch.randint(0, cfg.vocab_size, (args.batch_size, T + 1), device=device)
            try:
                torch.cuda.reset_peak_memory_stats()
                for i in range(args.bench_steps + 2):
                    if i == 2:  # first two steps are warmup
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(idx[:, :-1], idx[:, 1:])
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) / args.bench_steps * 1000
                peak = torch.cuda.max_memory_allocated() / 1e9
                tok_s = args.batch_size * T / ms * 1000
                label = f"chunk {chunk}" if chunk else "full"
                print(f"{T:>6} {label:>12} {ms:>9.1f} {tok_s:>9,.0f} {peak:>8.1f}")
            except torch.OutOfMemoryError:
                print(f"{T:>6} {('chunk ' + str(chunk)) if chunk else 'full':>12} {'out of memory':>28}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="Check chunked attention against full attention")
    p.add_argument("--bench", action="store_true", help="also time training steps (CUDA only)")
    p.add_argument("--n-embd", type=int, default=512, help="model width for --bench")
    p.add_argument("--vocab", type=int, default=32768, help="vocabulary size for --bench")
    p.add_argument("--batch-size", type=int, default=4, help="sequences per step for --bench")
    p.add_argument("--bench-blocks", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--bench-chunks", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--bench-steps", type=int, default=5)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    print("ChunkedAttention vs full attention (CPU, float64):")
    cpu = torch.device("cpu")
    ok = True
    for T, chunk in [(16, 4), (17, 4), (16, 16), (16, 32), (9, 1), (40, 7)]:
        ok &= check_function(2, 3, T, 12, 5, chunk, cpu)
    ok &= check(
        "gradcheck",
        0.0 if torch.autograd.gradcheck(
            lambda q, v: ChunkedAttention.apply(q, v, 3),
            (torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True),
             torch.randn(1, 1, 7, 3, dtype=torch.float64, requires_grad=True)),
        ) else 1.0,
    )

    print("\nSmall BDH model, loss and gradients (CPU, float32):")
    cfg = bdh.BDHConfig(n_layer=3, n_embd=64, n_head=4, dropout=0.0, mlp_internal_dim_multiplier=16, vocab_size=97)
    model = bdh.BDH(cfg)
    ok &= check_model(model, T=48, chunk=16, device=cpu)
    ok &= check_model(model, T=50, chunk=16, device=cpu)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        print("\nOn CUDA:")
        ok &= check_function(2, 4, 300, 64, 32, 64, cuda)
        model = model.to(cuda)
        ok &= check_model(model, T=48, chunk=16, device=cuda)
        ok &= check_model(model, T=256, chunk=64, device=cuda, autocast=True)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    if args.bench:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            bench(args)
        else:
            print("--bench needs a CUDA GPU")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
