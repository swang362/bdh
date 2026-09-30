# Copyright Pathway Technology, Inc.

"""Checks that recurrent inference (recurrent.py) matches BDH.forward.

Runs on the CPU in seconds with a small random model, in float32:

    python test_recurrent.py

Optionally also checks a real checkpoint (a slice of real text isn't needed:
random token ids exercise the same computation):

    python test_recurrent.py --checkpoint checkpoints/wiki_sp16384/best.pt

With a CUDA GPU it also checks CUDA graph replay (recurrent.py, cuda_graph=True).
"""

import argparse
import os
import tempfile

import torch

import bdh
from bdh import Attention
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

    # 4. beyond the window: prefill + steps and step-by-step agree with a slow,
    #    independent reference that sums the last window - 1 tokens explicitly
    w = min(32, max(2, window // 2))
    long_ids = torch.randint(0, vocab, (3 * w,), generator=gen).tolist()
    expected = reference_windowed(model, long_ids, w, device)
    rec = RecurrentBDH(model, window=w)
    stepped = torch.stack([rec.step(t) for t in long_ids])
    ok &= check(f"window {w}, {3 * w} tokens: step-by-step vs reference", rel_diff(stepped, expected))
    rec = RecurrentBDH(model, window=w)
    first = rec.prefill(long_ids[: 2 * w])
    rest = torch.stack([rec.step(t) for t in long_ids[2 * w :]])
    got = torch.cat([first[None], rest])
    ok &= check(f"window {w}, {3 * w} tokens: prefill + steps vs reference", rel_diff(got, expected[2 * w - 1 :]))

    # 5. CUDA graph replay equals normal execution (CUDA only)
    if device.type == "cuda":
        rec = RecurrentBDH(model, window=w, cuda_graph=True)
        graphed = torch.stack([rec.step(t) for t in long_ids])
        if rec.graph is None:
            print("  [FAIL] CUDA graph was not captured")
            ok = False
        else:
            ok &= check(f"CUDA graph vs reference, window {w}, {3 * w} tokens", rel_diff(graphed, expected))
            # reset and reuse the same recorded graph, as chat.py does across turns
            first = rec.prefill(long_ids[:w])
            rest = torch.stack([rec.step(t) for t in long_ids[w:]])
            got = torch.cat([first[None], rest])
            ok &= check("CUDA graph reused after reset", rel_diff(got, expected[w - 1 :]))
            # loading a saved state into a graphed state keeps the graph valid
            src = RecurrentBDH(model, window=w)
            src.prefill(long_ids[:w])
            path = os.path.join(tempfile.mkdtemp(), "state.pt")
            src.save(path)
            rec.load(path)
            got = torch.stack([rec.step(t) for t in long_ids[w:]])
            ok &= check("CUDA graph after load", rel_diff(got, expected[w:]))

    # 6. save and load: continuing a loaded state equals continuing without saving
    rec = RecurrentBDH(model, window=w)
    rec.prefill(long_ids[:w])
    for t in long_ids[w : 2 * w]:
        rec.step(t)
    path = os.path.join(tempfile.mkdtemp(), "state.pt")
    rec.save(path, extra={"note": "test"})
    loaded = RecurrentBDH(model, window=w)
    extra = loaded.load(path)
    a = torch.stack([rec.step(t) for t in long_ids[2 * w :]])
    b = torch.stack([loaded.step(t) for t in long_ids[2 * w :]])
    ok &= check("save + load, then continue", rel_diff(b, a))
    ok &= check("  ... and vs reference", rel_diff(b, expected[2 * w :]))
    ok &= report("save stores extra data", extra == {"note": "test"})

    # 7. snapshot and restore: stepping from a restored snapshot repeats exactly
    snap = rec.snapshot()
    first = rec.step(long_ids[0])
    rec.restore(snap)
    ok &= check("snapshot + restore", rel_diff(rec.step(long_ids[0]), first))

    # 8. a state can't be loaded into a different model or window size
    other = bdh.BDH(model.config).to(device).eval()  # same config, different random weights
    ok &= report("load rejects a different model", raises(ValueError, RecurrentBDH(other, window=w).load, path))
    ok &= report("load rejects a different window", raises(ValueError, RecurrentBDH(model, window=w + 1).load, path))
    return ok


def report(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return ok


def raises(error, fn, *args):
    try:
        fn(*args)
    except error:
        return True
    return False


@torch.no_grad()
def reference_windowed(model, ids, window, device):
    """Slow reference for the sliding window: every position explicitly sums the
    contributions of the last window - 1 tokens, kept in plain Python lists."""
    m = model
    cap = window - 1
    freqs = m.attn.freqs.view(-1).double().cpu()
    entries = [[] for _ in range(m.config.n_layer)]
    out = []
    for pos, token in enumerate(ids):
        phases = ((pos * freqs) % 1).float().to(device)
        x = m.ln(m.embed.weight[token])
        for layer in range(m.config.n_layer):
            x_sparse = torch.relu(torch.einsum("d,hdn->hn", x, m.encoder))
            q = Attention.rope(phases, x_sparse)
            y = torch.zeros(m.config.n_head, m.config.n_embd, device=device)
            for kq, kv in entries[layer][-cap:]:
                y += (q * kq).sum(-1, keepdim=True) * kv
            entries[layer].append((q, x))
            y_sparse = torch.relu(torch.einsum("hd,hdn->hn", m.ln(y), m.encoder_v))
            x = m.ln(x + m.ln((x_sparse * y_sparse).reshape(-1) @ m.decoder))
        out.append(x @ m.lm_head)
    return torch.stack(out)


def main():
    p = argparse.ArgumentParser(description="Check recurrent inference against BDH.forward")
    p.add_argument("--checkpoint", help="also check this checkpoint (runs in float32)")
    p.add_argument("--tokens", type=int, default=64, help="sequence length for the checkpoint check")
    p.add_argument("--cpu", action="store_true", help="run the checkpoint check on CPU")
    args = p.parse_args()
    torch.manual_seed(0)
    # float32 matmuls, so differences reflect the method, not bf16/TF32 rounding
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Small random model (CPU, float32):")
    cfg = bdh.BDHConfig(n_layer=3, n_embd=64, n_head=4, dropout=0.0, mlp_internal_dim_multiplier=16, vocab_size=97)
    small = bdh.BDH(cfg).eval()
    ok = run_checks(small, T=48, window=48, device=torch.device("cpu"))
    if torch.cuda.is_available() and not args.cpu:
        print("\nSmall random model (CUDA, float32, includes the CUDA graph checks):")
        ok &= run_checks(small.to("cuda"), T=48, window=48, device=torch.device("cuda"))

    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = bdh.BDH(bdh.BDHConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        window = checkpoint.get("block_size", 512)
        print(f"\n{args.checkpoint} ({device}, float32, window {window}):")
        ok &= run_checks(model, T=min(args.tokens, window), window=window, device=device)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
