# Copyright Pathway Technology, Inc.

"""Checks that recurrent inference (recurrent.py) matches BDH.forward.

Runs on the CPU in seconds with a small random model, in float32:

    python test_recurrent.py

Optionally also checks a real checkpoint (a slice of real text isn't needed:
random token ids exercise the same computation):

    python test_recurrent.py --checkpoint checkpoints/wiki_sp16384/best.pt
"""

import argparse

import torch

import bdh
from recurrent import RecurrentBDH

TOLERANCE = 1e-3  # max |difference| of logits, relative to the largest logit


def rel_diff(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def check(name, diff):
    ok = diff < TOLERANCE
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: max relative difference {diff:.2e}")
    return ok


@torch.no_grad()
def run_checks(model, T, window, device):
    """T: sequence length for the checks; window: sliding window (training block size)."""
    vocab = model.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (T,), generator=gen).tolist()
    ref = model(torch.tensor([ids], device=device))[0][0]  # (T, vocab), parallel forward
    ok = True

    # 1. feeding tokens one at a time equals the parallel forward pass
    rec = RecurrentBDH(model, window=None)
    steps = torch.stack([rec.step(t) for t in ids])
    ok &= check(f"step-by-step vs forward, {T} tokens", rel_diff(steps, ref))

    # 2. parallel prefill, then steps, equals the forward pass
    half = T // 2
    rec = RecurrentBDH(model, window=None)
    first = rec.prefill(ids[:half])
    rest = torch.stack([rec.step(t) for t in ids[half:]])
    ok &= check("prefill + steps vs forward", rel_diff(torch.cat([first[None], rest]), ref[half - 1 :]))

    # 3. with a sliding window, results are identical while everything fits in it
    rec = RecurrentBDH(model, window=T)
    windowed = torch.stack([rec.step(t) for t in ids])
    ok &= check(f"window {T} vs forward (fits in window)", rel_diff(windowed, ref))

    # 4. beyond the window: prefill and step-by-step agree, and the state never
    #    holds more than window - 1 tokens
    w = max(2, window // 2)
    long_ids = torch.randint(0, vocab, (3 * w,), generator=gen).tolist()
    rec_a = RecurrentBDH(model, window=w)
    a = rec_a.prefill(long_ids)
    rec_b = RecurrentBDH(model, window=w)
    for t in long_ids:
        b = rec_b.step(t)
    ok &= check(f"window {w}, {3 * w} tokens: prefill vs step-by-step", rel_diff(a, b))
    held = max(len(e) for e in rec_b.entries)
    ok &= held <= w - 1
    print(f"  [{'OK' if held <= w - 1 else 'FAIL'}] window {w}: state holds {held} tokens (max {w - 1})")
    return ok


def main():
    p = argparse.ArgumentParser(description="Check recurrent inference against BDH.forward")
    p.add_argument("--checkpoint", help="also check this checkpoint (runs in float32)")
    p.add_argument("--tokens", type=int, default=64, help="sequence length for the checkpoint check")
    p.add_argument("--cpu", action="store_true", help="run the checkpoint check on CPU")
    args = p.parse_args()
    torch.manual_seed(0)

    print("Small random model (CPU, float32):")
    cfg = bdh.BDHConfig(n_layer=3, n_embd=64, n_head=4, dropout=0.0, mlp_internal_dim_multiplier=16, vocab_size=97)
    ok = run_checks(bdh.BDH(cfg).eval(), T=48, window=48, device=torch.device("cpu"))

    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = bdh.BDH(bdh.BDHConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        # float32 matmuls, so differences reflect the method, not bf16/TF32 rounding
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        window = checkpoint.get("block_size", 512)
        print(f"\n{args.checkpoint} ({device}, float32, window {window}):")
        ok &= run_checks(model, T=min(args.tokens, window), window=window, device=device)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
