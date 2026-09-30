# Copyright Pathway Technology, Inc.

"""Fact-recall probe: greedy completions of factual prompts, scored against answers.

Gives a simple, explainable accuracy number to track alongside validation bpb,
e.g. across the snapshots of one run or between model sizes.

Examples:
    python probe.py checkpoints/wiki_sp16384/best.pt
    python probe.py checkpoints/wiki_sp16384 --verbose
    python probe.py checkpoints/wiki_sp16384 checkpoints/wiki_d384 --csv probe.csv
    python probe.py checkpoints/wiki_sp16384/best.pt --probes my_probes.json
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from contextlib import nullcontext

import bdh
import tokenizer as tokenizers
import torch

from inference import token_stream

# (prompt, accepted answers). A probe is correct if any answer appears as a whole
# word (case-insensitive) in the greedy continuation. Most prompts follow
# Wikipedia's phrasing, since a pretrained-only model continues text rather than
# answering questions.
DEFAULT_PROBES = [
    ("Paris is the capital of", ["France"]),
    ("The capital of France is", ["Paris"]),
    ("Berlin is the capital of", ["Germany"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("Tokyo is the capital of", ["Japan"]),
    ("Rome is the capital of", ["Italy"]),
    ("Madrid is the capital of", ["Spain"]),
    ("The capital of Russia is", ["Moscow"]),
    ("London is the capital of", ["England", "United Kingdom", "UK", "Britain"]),
    ("Canberra is the capital of", ["Australia"]),
    ("Ottawa is the capital of", ["Canada"]),
    ("World War II ended in", ["1945"]),
    ("World War I began in", ["1914"]),
    ("The French Revolution began in", ["1789"]),
    ("The United States Declaration of Independence was adopted in", ["1776"]),
    ("Christopher Columbus reached the Americas in", ["1492"]),
    ("The Berlin Wall fell in", ["1989"]),
    ("Albert Einstein was born in", ["Ulm", "1879"]),
    ("William Shakespeare was born in", ["Stratford", "1564"]),
    ("Wolfgang Amadeus Mozart was born in", ["Salzburg", "1756"]),
    ("Napoleon Bonaparte was born in", ["Corsica", "Ajaccio", "1769"]),
    ("Jupiter is the largest", ["planet"]),
    ("The largest planet in the Solar System is", ["Jupiter"]),
    ("The closest planet to the Sun is", ["Mercury"]),
    ("The Earth orbits the", ["Sun"]),
    ("Water is composed of hydrogen and", ["oxygen"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The chemical symbol for iron is", ["Fe"]),
    ("The boiling point of water at sea level is", ["100"]),
    ("The Amazon River is in", ["South America", "Brazil", "Peru"]),
    ("The Nile is a river in", ["Africa", "Egypt"]),
    ("Mount Everest is the highest mountain in", ["world", "Earth", "Asia", "Himalayas"]),
    ("The Pacific Ocean is the largest", ["ocean"]),
    ("The Great Wall is located in", ["China"]),
    ("The Eiffel Tower is located in", ["Paris"]),
    ("The Statue of Liberty is located in", ["New York"]),
    ("The official language of Brazil is", ["Portuguese"]),
    ("The official language of Mexico is", ["Spanish"]),
    ("Isaac Newton was an English", ["physicist", "mathematician", "scientist", "astronomer"]),
    ("Charles Darwin is best known for", ["evolution", "natural selection"]),
    ("The Mona Lisa was painted by", ["Leonardo", "da Vinci"]),
    ("Romeo and Juliet was written by", ["Shakespeare"]),
    ("The first President of the United States was", ["Washington"]),
    ("Barack Obama was the 44th President of the", ["United States"]),
    ("The currency of Japan is the", ["yen"]),
    ("The currency of the United Kingdom is the", ["pound"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The human heart has four", ["chambers"]),
    ("There are seven days in a", ["week"]),
    ("The Roman Empire was ruled from the city of", ["Rome", "Constantinople"]),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Score fact recall of BDH checkpoints with greedy completions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "checkpoints",
        nargs="+",
        help="checkpoint files, or directories (every .pt file inside is probed)",
    )
    p.add_argument(
        "--probes",
        help='JSON file with a list of {"prompt": ..., "answers": [...]} (default: built-in list)',
    )
    p.add_argument("--max-new-tokens", type=int, default=10, help="tokens generated per probe")
    p.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="context window (default: the training block size saved in the checkpoint)",
    )
    p.add_argument("--verbose", action="store_true", help="print every completion")
    p.add_argument(
        "--recurrent",
        action="store_true",
        help="generate recurrently (docs/recurrent.md); scores should match the default method",
    )
    p.add_argument("--csv", help="also write the per-checkpoint results to this CSV file")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    return p.parse_args()


def load_probes(path):
    if path is None:
        return DEFAULT_PROBES
    with open(path, encoding="utf-8") as f:
        return [(p["prompt"], p["answers"]) for p in json.load(f)]


def find_checkpoints(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "*.pt")))
        elif os.path.exists(path):
            files.append(path)
        else:
            raise SystemExit(f"Not found: {path}")
    return files


def is_correct(completion, answers):
    return any(
        re.search(r"\b" + re.escape(a) + r"\b", completion, re.IGNORECASE) for a in answers
    )


def probe_checkpoint(path, probes, args, device, ctx):
    # load on CPU: checkpoints also hold optimizer state, which isn't needed here
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = bdh.BDH(bdh.BDHConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tok = tokenizers.from_state(checkpoint.get("tokenizer"))
    context_size = args.context_size
    if context_size is None:
        context_size = checkpoint.get("block_size", 512)
    step = checkpoint["step"]
    del checkpoint

    results = []
    for prompt, answers in probes:
        ids = tok.encode(prompt)
        with ctx:
            # top_k=1 is greedy decoding: deterministic, the model's single best guess
            new = list(
                token_stream(
                    model, ids, device, args.recurrent, args.max_new_tokens, 1.0, 1, context_size or None
                )
            )
        # decode the whole sequence and cut the prompt, so text spanning the
        # boundary decodes correctly; stop at the end-of-text marker
        full = tok.decode(ids + new)
        completion = full[len(tok.decode(ids)):].split(tokenizers.EOT)[0]
        results.append((prompt, answers, completion, is_correct(completion, answers)))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return step, results


def main():
    args = parse_args()
    sys.stdout.reconfigure(errors="backslashreplace")
    probes = load_probes(args.probes)
    files = find_checkpoints(args.checkpoints)
    if not files:
        raise SystemExit("No checkpoints found")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_cuda else nullcontext()
    print(f"Probing {len(files)} checkpoint(s) with {len(probes)} probes on {device}")

    summary = []
    for path in files:
        step, results = probe_checkpoint(path, probes, args, device, ctx)
        correct = sum(r[3] for r in results)
        summary.append((path, step, correct, len(results)))
        print(f"\n{path} (step {step}): {correct}/{len(results)} correct ({100 * correct / len(results):.1f}%)")
        if args.verbose:
            for prompt, answers, completion, ok in results:
                one_line = " ".join(completion.split())
                print(f"  [{'OK' if ok else '--'}] {prompt} -> {one_line!r}  (expected: {' / '.join(answers)})")

    if len(summary) > 1:
        print("\nSummary (by training step):")
        for path, step, correct, total in sorted(summary, key=lambda s: (s[1], s[0])):
            print(f"  step {step:>8}  {correct:>3}/{total}  {100 * correct / total:5.1f}%  {os.path.basename(path)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["checkpoint", "step", "correct", "total", "accuracy"])
            for path, step, correct, total in summary:
                w.writerow([path, step, correct, total, f"{correct / total:.4f}"])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
