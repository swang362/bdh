# Copyright Pathway Technology, Inc.

"""Supervised fine-tuning (SFT) of a pretrained BDH checkpoint for chat / QA.

Examples are rendered with chat_format.py; only the assistant answers are
trained on (prompt tokens get target -100, which cross-entropy ignores).

Data (--data, one or more; mixed together):
    dolly           databricks-dolly-15k: 15K human-written instruction/answer pairs
    alpaca          Stanford Alpaca: 52K generated instruction/answer pairs
    FILE.jsonl      your own data, one JSON object per line, either
                      {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}, ...]}
                    or
                      {"prompt": ..., "response": ..., "context": optional, "system": optional}
                    (also accepted: instruction/question for the prompt,
                     output/answer for the response, input for the context)
    FILE.json       the same records as a JSON list

Examples:
    python finetune.py --base checkpoints/wiki_sp16384/best.pt --data dolly
    python finetune.py --base checkpoints/wiki_sp16384/best.pt --data dolly my_kb_qa.jsonl \\
        --out-dir checkpoints/wiki_chat --epochs 3
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import bdh
import torch

import chat_format
import tokenizer as tokenizers
from prepare_data import download

ROOT = os.path.dirname(os.path.abspath(__file__))
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a BDH checkpoint for chat / question answering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base", required=True, help="pretrained checkpoint to start from")
    p.add_argument(
        "--data",
        nargs="+",
        default=["dolly"],
        help="datasets to mix: dolly, alpaca, and/or .jsonl/.json files",
    )
    p.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "chat"), help="output checkpoint directory")
    p.add_argument("--system", default=None, help="system prompt added to every example that has none")
    p.add_argument("--val-fraction", type=float, default=0.02, help="share of examples held out for validation")
    p.add_argument("--epochs", type=float, default=3, help="passes over the training examples")
    p.add_argument("--max-iters", type=int, default=None, help="total steps (overrides --epochs)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=None, help="max tokens per example (default: the base checkpoint's block size)")
    p.add_argument("--lr", type=float, default=1e-4, help="peak learning rate (lower than pretraining)")
    p.add_argument("--min-lr", type=float, default=None, help="final lr of the cosine schedule (default: lr / 10)")
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max gradient norm (0 disables)")
    p.add_argument("--dropout", type=float, default=None, help="override the base model's dropout (e.g. 0.1 for small datasets)")
    p.add_argument("--eval-freq", type=int, default=200, help="evaluate every N steps (and at the end)")
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from <out-dir>/latest.pt if it exists",
    )
    p.add_argument("--sample-prompt", default="What is the capital of France?", help="question answered after training ('' to skip)")
    p.add_argument("--cpu", action="store_true", help="train on CPU")
    return p.parse_args()


# ---------------------------------------------------------------- data


def record_to_messages(rec, system):
    """Convert one data record to chat messages, or None if it's unusable."""
    if "messages" in rec:
        messages = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    else:
        prompt = rec.get("prompt") or rec.get("instruction") or rec.get("question")
        response = rec.get("response") or rec.get("output") or rec.get("answer")
        if not prompt or not response:
            return None
        context = rec.get("context") or rec.get("input")
        user = prompt.strip() + (f"\n\n{context.strip()}" if context and context.strip() else "")
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        if rec.get("system"):
            messages.insert(0, {"role": "system", "content": rec["system"]})
    if system and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system})
    if not any(m["role"] == "assistant" and m["content"].strip() for m in messages):
        return None
    return messages


def read_records(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(name, raw_dir):
    if name == "dolly":
        path = os.path.join(raw_dir, "dolly", "databricks-dolly-15k.jsonl")
        download(DOLLY_URL, path)
        return read_records(path)
    if name == "alpaca":
        path = os.path.join(raw_dir, "alpaca", "alpaca_data.json")
        download(ALPACA_URL, path)
        return read_records(path)
    if os.path.exists(name):
        return read_records(name)
    raise SystemExit(f"Unknown dataset or file not found: {name}")


def build_examples(tok, conversations, max_len):
    """Tokenize conversations to (ids, mask), truncated to max_len tokens."""
    examples, truncated, skipped = [], 0, 0
    for messages in conversations:
        ids, mask = chat_format.encode(tok, messages)
        if len(ids) > max_len:
            ids, mask = ids[:max_len], mask[:max_len]
            truncated += 1
        # targets are ids[1:], so at least one trained token must follow the first
        if not any(mask[1:]):
            skipped += 1
            continue
        examples.append((ids, mask))
    return examples, truncated, skipped


def make_batch(examples, device):
    """Right-pad to the longest example. Padding and prompt targets are -100 (ignored)."""
    length = max(len(ids) for ids, _ in examples) - 1
    x = torch.zeros((len(examples), length), dtype=torch.long)
    y = torch.full((len(examples), length), -100, dtype=torch.long)
    for b, (ids, mask) in enumerate(examples):
        n = len(ids) - 1
        x[b, :n] = torch.tensor(ids[:-1])
        targets = torch.tensor(ids[1:])
        trained = torch.tensor(mask[1:])
        y[b, :n] = torch.where(trained, targets, torch.full_like(targets, -100))
    # attention is causal, so right padding never affects the real tokens
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- training


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_cuda and dtype == torch.float16)
    torch.manual_seed(args.seed)

    latest_path = os.path.join(args.out_dir, "latest.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    resuming = args.resume and os.path.exists(latest_path)
    source = latest_path if resuming else args.base
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)

    config = dict(checkpoint["config"])
    if args.dropout is not None:
        config["dropout"] = args.dropout
    model = bdh.BDH(bdh.BDHConfig(**config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    tok = tokenizers.from_state(checkpoint.get("tokenizer"))
    block_size = args.block_size or checkpoint.get("block_size", 512)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{'Resuming' if resuming else 'Fine-tuning'} from {source} (step {checkpoint['step']}): "
        f"{n_params:,} params, {tok.type} tokenizer (vocab {tok.vocab_size}), block {block_size}, {device}"
    )

    # data: mix all sources, shuffle once with a fixed seed, hold out a validation share
    raw_dir = os.path.join(ROOT, "data", "raw")
    conversations = []
    for name in args.data:
        records = load_dataset(name, raw_dir)
        convs = [c for c in (record_to_messages(r, args.system) for r in records) if c]
        print(f"  {name}: {len(convs):,} conversations")
        conversations += convs
    random.Random(args.seed).shuffle(conversations)
    examples, truncated, skipped = build_examples(tok, conversations, block_size + 1)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    trained_tokens = sum(sum(m[1:]) for _, m in train_examples)
    print(
        f"  {len(train_examples):,} train / {len(val_examples):,} val examples, "
        f"{trained_tokens:,} trained answer tokens ({truncated:,} truncated, {skipped:,} skipped)"
    )

    steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    max_iters = args.max_iters or max(1, int(args.epochs * steps_per_epoch))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

    def get_lr(step):
        if step < args.warmup_iters:
            return args.lr * (step + 1) / args.warmup_iters
        progress = min(1.0, (step - args.warmup_iters) / max(1, max_iters - args.warmup_iters))
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step, best_val_loss = 0, float("inf")
    if resuming:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    base_tokenizer_state = checkpoint.get("tokenizer")
    base_info = checkpoint.get("chat", {}).get("base", args.base)
    del checkpoint

    def save(path, step):
        os.makedirs(args.out_dir, exist_ok=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config,
            "block_size": block_size,
            "tokenizer": base_tokenizer_state,
            "best_val_loss": best_val_loss,
            "chat": {"format": chat_format.FORMAT_VERSION, "base": base_info, "data": args.data, "system": args.system},
        }
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    @torch.no_grad()
    def evaluate():
        # loss per trained token over all validation examples, dropout off
        model.eval()
        total, count = 0.0, 0
        for i in range(0, len(val_examples), args.batch_size):
            x, y = make_batch(val_examples[i : i + args.batch_size], device)
            with ctx:
                _, loss = model(x, y)
            n = (y != -100).sum().item()
            total += loss.item() * n
            count += n
        model.train()
        return total / max(1, count)

    def run_eval(step):
        nonlocal best_val_loss
        val_loss = evaluate()
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            save(best_path, step)
        print(
            f"Eval step {step}: val loss {val_loss:.4f} (ppl {math.exp(val_loss):.1f}) | "
            + (f"new best, saved {best_path}" if improved else f"best {best_val_loss:.4f}")
        )

    def epoch_order(epoch):
        # deterministic per epoch, so a resumed run continues the same order
        return torch.randperm(len(train_examples), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()

    print(f"Training {max_iters} steps ({max_iters / steps_per_epoch:.2f} epochs), lr {args.lr} -> {min_lr}")
    model.train()
    step = start_step
    loss_acc, loss_n, t0 = 0.0, 0, time.perf_counter()
    try:
        while step < max_iters:
            epoch, offset = divmod(step, steps_per_epoch)
            order = epoch_order(epoch)
            batch_ids = order[offset * args.batch_size : (offset + 1) * args.batch_size]
            x, y = make_batch([train_examples[i] for i in batch_ids], device)
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            loss_acc += loss.item()
            loss_n += 1
            if step % args.log_freq == 0 or step == max_iters:
                dt = time.perf_counter() - t0
                print(f"Step {step}/{max_iters} (epoch {step / steps_per_epoch:.2f}) loss {loss_acc / loss_n:.4f} | lr {lr:.2e} | {1000 * dt / loss_n:.0f} ms/step")
                loss_acc, loss_n, t0 = 0.0, 0, time.perf_counter()
            if step % args.eval_freq == 0:
                run_eval(step)
                save(latest_path, step)
    except KeyboardInterrupt:
        print("Interrupted, saving checkpoint")
        save(latest_path, step)
        raise
    if step % args.eval_freq != 0:
        run_eval(step)
    save(latest_path, step)
    print(f"Done. Best checkpoint: {best_path} (val loss {best_val_loss:.4f})")

    if args.sample_prompt:
        from chat import generate_reply

        model.eval()
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": args.sample_prompt})
        print(f"\nUser: {args.sample_prompt}\nAssistant: ", end="")
        with ctx:
            generate_reply(model, tok, messages, block_size, on_text=lambda t: print(t, end="", flush=True))
        print()


if __name__ == "__main__":
    main()
