# Copyright Pathway Technology, Inc.

import argparse
import dataclasses
import math
import os
import shutil
import time
from contextlib import nullcontext

import bdh
import numpy as np
import tokenizer as tokenizers
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration (defaults; can be overridden from the command line, see parse_args)
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
GRAD_ACCUM = 1  # micro-batches per optimizer step; each holds BATCH_SIZE // GRAD_ACCUM sequences
MICRO_BATCH_SIZE = BATCH_SIZE // GRAD_ACCUM
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
LR_SCHEDULE = "cosine"  # "cosine" (decay to MIN_LR) or "constant"
WARMUP_ITERS = 200  # linear warmup from ~0 to LEARNING_RATE
MIN_LR = LEARNING_RATE / 10  # final lr at MAX_ITERS for the cosine schedule
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CKPT_FREQ = 500  # save a checkpoint every CKPT_FREQ steps
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "latest.pt")
SNAPSHOT_FREQ = 5000  # keep a copy of the checkpoint every SNAPSHOT_FREQ steps (0 disables)
EVAL_FREQ = 500  # evaluate on the validation split every EVAL_FREQ steps (0 disables)
EVAL_ITERS = 50  # validation batches averaged per evaluation
BEST_PATH = os.path.join(CKPT_DIR, "best.pt")  # checkpoint with the lowest validation loss
best_val_loss = float("inf")
RESUME = True  # resume from CKPT_PATH if it exists

DEFAULT_INPUT_PATH = os.path.join(os.path.dirname(__file__), "input.txt")
input_file_path = DEFAULT_INPUT_PATH
data_dir = None  # directory with train.bin / val.bin from prepare_data.py
TOKENIZER = tokenizers.ByteTokenizer()  # replaced by the data dir's tokenizer, see __main__
DATA_DTYPE = np.uint8  # dtype of the token ids in train.bin / val.bin
BYTES_PER_TOKEN = 1.0  # average text bytes per token, for reporting bits per byte


def parse_args():
    d = bdh.BDHConfig()
    p = argparse.ArgumentParser(
        description="Train a BDH model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("training")
    g.add_argument("--max-iters", type=int, default=MAX_ITERS, help="total training steps")
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="sequences per optimizer step")
    g.add_argument(
        "--grad-accum",
        type=int,
        default=GRAD_ACCUM,
        help="split each step into N micro-batches of batch-size / N sequences: same tokens per step, "
        "about N times less activation memory; batch-size must be divisible by N",
    )
    g.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="sequence length in tokens")
    g.add_argument("--lr", type=float, default=LEARNING_RATE, help="peak AdamW learning rate")
    g.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default=LR_SCHEDULE,
        help="after warmup: cosine decay to --min-lr at --max-iters, or keep --lr constant",
    )
    g.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS, help="linear lr warmup steps")
    g.add_argument("--min-lr", type=float, default=None, help="final lr of the cosine schedule; None means lr / 10")
    g.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    g.add_argument("--seed", type=int, default=1337, help="random seed (ignored when resuming)")
    g.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use torch.compile",
    )
    g.add_argument(
        "--data",
        default=DEFAULT_INPUT_PATH,
        help="training text file, read as raw bytes and split 90/10 (Tiny Shakespeare is downloaded if the default is missing); ignored when --data-dir is set",
    )
    g.add_argument(
        "--data-dir",
        default=None,
        help="directory with train.bin / val.bin made by prepare_data.py",
    )
    g = p.add_argument_group("model")
    g.add_argument("--n-layer", type=int, default=d.n_layer, help="number of layers (weights are shared across layers)")
    g.add_argument("--n-embd", type=int, default=d.n_embd, help="embedding dimension D")
    g.add_argument("--n-head", type=int, default=d.n_head, help="number of heads")
    g.add_argument("--dropout", type=float, default=d.dropout, help="dropout rate")
    g.add_argument("--mlp-mult", type=int, default=d.mlp_internal_dim_multiplier, help="sparse dim N = mlp_mult * n_embd / n_head")
    g = p.add_argument_group("logging and checkpoints")
    g.add_argument("--log-freq", type=int, default=LOG_FREQ, help="log every N steps")
    g.add_argument("--ckpt-freq", type=int, default=CKPT_FREQ, help="save a checkpoint every N steps")
    g.add_argument("--ckpt-dir", default=CKPT_DIR, help="checkpoint directory")
    g.add_argument(
        "--snapshot-freq",
        type=int,
        default=SNAPSHOT_FREQ,
        help="every N steps (and at the end) also keep a copy named step<N>_bpb<B>.pt (training bits per byte); 0 disables",
    )
    g.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=RESUME,
        help="resume from <ckpt-dir>/latest.pt if it exists",
    )
    g = p.add_argument_group("validation")
    g.add_argument(
        "--eval-freq",
        type=int,
        default=EVAL_FREQ,
        help="evaluate on the validation split every N steps (and at the end), saving <ckpt-dir>/best.pt on improvement; 0 disables",
    )
    g.add_argument("--eval-iters", type=int, default=EVAL_ITERS, help="validation batches averaged per evaluation")
    g = p.add_argument_group("sample after training")
    g.add_argument("--prompt", default="To be or ", help="prompt for the sample generated after training")
    g.add_argument("--sample-tokens", type=int, default=100, help="number of tokens to generate (0 to skip sampling)")
    return p.parse_args()


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        if input_file_path != DEFAULT_INPUT_PATH:
            raise SystemExit(f"Data file not found: {input_file_path}")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split, generator=None):
    if data_dir is not None:
        data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=DATA_DTYPE, mode="r")
    else:
        # plain text file: treat the file as bytes
        data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
        if split == "train":
            data = data[: int(0.9 * len(data))]
        else:
            data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (MICRO_BATCH_SIZE,), generator=generator)
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def save_checkpoint(model, optimizer, step, path=None):
    # model must be the uncompiled module so state_dict keys have no "_orig_mod." prefix
    path = path or CKPT_PATH
    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,  # number of completed steps, i.e. the next step to run
        "config": dataclasses.asdict(BDH_CONFIG),
        "block_size": BLOCK_SIZE,  # training context length, used by inference.py
        "tokenizer": tokenizers.to_state(TOKENIZER),
        "best_val_loss": best_val_loss,  # so a resumed run keeps improving on the same best
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    # write to a temp file first so an interrupted save never corrupts the last checkpoint
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved checkpoint at step {step} to {path}")


def snapshot_checkpoint(step, loss):
    # copy latest.pt to a permanent, descriptive name that later saves won't overwrite;
    # the name uses bits per byte, which is comparable across tokenizers (loss per token isn't)
    name = f"step{step:07d}"
    if loss is not None:
        name += f"_bpb{loss / math.log(2) / BYTES_PER_TOKEN:.4f}"
    path = os.path.join(CKPT_DIR, name + ".pt")
    shutil.copyfile(CKPT_PATH, path)
    print(f"Saved snapshot {path}")


def load_checkpoint(model, optimizer):
    global best_val_loss
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    # checkpoints from before tokenizer support have no entry and are byte-level
    ckpt_tokenizer = tokenizers.from_state(checkpoint.get("tokenizer"))
    if not tokenizers.same_tokenizer(ckpt_tokenizer, TOKENIZER):
        raise ValueError(
            f"Checkpoint {CKPT_PATH} uses a different tokenizer "
            f"({ckpt_tokenizer.type}, vocab {ckpt_tokenizer.vocab_size}) than the data "
            f"({TOKENIZER.type}, vocab {TOKENIZER.vocab_size}); use another --ckpt-dir or --no-resume"
        )
    if checkpoint["config"] != dataclasses.asdict(BDH_CONFIG):
        raise ValueError(
            f"Checkpoint config {checkpoint['config']} does not match "
            f"current config {dataclasses.asdict(BDH_CONFIG)}"
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # load_state_dict restores the saved lr/weight decay; use the current settings instead
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE
        group["weight_decay"] = WEIGHT_DECAY
    scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["rng_cpu"].cpu())
    if torch.cuda.is_available() and checkpoint["rng_cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_cuda"]])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resumed from checkpoint {CKPT_PATH} at step {checkpoint['step']}")
    return checkpoint["step"]


def get_lr(step):
    # linear warmup, then cosine decay from LEARNING_RATE to MIN_LR at MAX_ITERS
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS
    if LR_SCHEDULE == "constant":
        return LEARNING_RATE
    if step >= MAX_ITERS:
        return MIN_LR
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


def sync():
    # wait for queued GPU work so wall-clock timings are accurate
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model):
    """Average loss over EVAL_ITERS validation batches, with dropout off."""
    # a fixed seed gives the same validation batches every time, so evaluations are
    # comparable with each other, and the training RNG stream is left untouched
    generator = torch.Generator().manual_seed(0)
    model.eval()
    total = 0.0
    # batches are micro-batches, so run GRAD_ACCUM times as many to cover the same tokens
    n_batches = EVAL_ITERS * GRAD_ACCUM
    for _ in range(n_batches):
        x, y = get_batch("val", generator=generator)
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


if __name__ == "__main__":
    args = parse_args()
    # override the module-level configuration used by the functions above
    input_file_path = args.data
    data_dir = args.data_dir
    if data_dir is None:
        fetch_data()
    else:
        for split in ("train", "val"):
            path = os.path.join(data_dir, f"{split}.bin")
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found (run prepare_data.py first)")
        meta = tokenizers.load_meta(data_dir)
        TOKENIZER = tokenizers.from_data_dir(data_dir)
        DATA_DTYPE = np.dtype(meta.get("dtype", "uint8"))
        BYTES_PER_TOKEN = meta.get("bytes_per_token", 1.0)
        print(
            f"Using dataset {data_dir} ({TOKENIZER.type} tokenizer, "
            f"vocab {TOKENIZER.vocab_size}, {BYTES_PER_TOKEN:.2f} bytes/token)"
        )
    BDH_CONFIG = bdh.BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        dropout=args.dropout,
        mlp_internal_dim_multiplier=args.mlp_mult,
        vocab_size=TOKENIZER.vocab_size,
    )
    BLOCK_SIZE = args.block_size
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    if GRAD_ACCUM < 1 or BATCH_SIZE % GRAD_ACCUM != 0:
        raise SystemExit(f"--batch-size {BATCH_SIZE} must be divisible by --grad-accum {GRAD_ACCUM}")
    MICRO_BATCH_SIZE = BATCH_SIZE // GRAD_ACCUM
    MAX_ITERS = args.max_iters
    LEARNING_RATE = args.lr
    LR_SCHEDULE = args.lr_schedule
    WARMUP_ITERS = args.warmup_iters
    MIN_LR = args.min_lr if args.min_lr is not None else args.lr / 10
    WEIGHT_DECAY = args.weight_decay
    LOG_FREQ = args.log_freq
    CKPT_FREQ = args.ckpt_freq
    CKPT_DIR = args.ckpt_dir
    CKPT_PATH = os.path.join(CKPT_DIR, "latest.pt")
    SNAPSHOT_FREQ = args.snapshot_freq
    EVAL_FREQ = args.eval_freq
    EVAL_ITERS = args.eval_iters
    BEST_PATH = os.path.join(CKPT_DIR, "best.pt")
    RESUME = args.resume
    torch.manual_seed(args.seed)

    raw_model = bdh.BDH(BDH_CONFIG).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    if GRAD_ACCUM > 1:
        print(
            f"Gradient accumulation: {GRAD_ACCUM} micro-batches of {MICRO_BATCH_SIZE} "
            f"= {BATCH_SIZE} sequences ({BATCH_SIZE * BLOCK_SIZE:,} tokens) per step"
        )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    start_step = 0
    if RESUME and os.path.exists(CKPT_PATH):
        start_step = load_checkpoint(raw_model, optimizer)
    model = torch.compile(raw_model) if args.compile else raw_model

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    avg_loss = None
    completed_steps = start_step
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    sync()
    train_start = time.perf_counter()
    window_start = train_start

    def recent_loss():
        # average training loss since the last log line, or of the last log window
        if loss_steps > 0:
            return loss_acc.item() / loss_steps
        return avg_loss

    def run_eval(step):
        # evaluate on the validation split; keep best.pt at the lowest validation loss
        global best_val_loss
        val_loss = evaluate(model)
        val_bpb = val_loss / math.log(2) / BYTES_PER_TOKEN
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        best_bpb = best_val_loss / math.log(2) / BYTES_PER_TOKEN
        print(
            f"Eval step {step}: val loss {val_loss:.4f} ({val_bpb:.4f} bpb) | "
            + ("new best" if improved else f"best {best_val_loss:.4f} ({best_bpb:.4f} bpb)")
        )
        if improved:
            save_checkpoint(raw_model, optimizer, step, BEST_PATH)

    try:
        for step in range(start_step, MAX_ITERS):
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            # gradient accumulation: GRAD_ACCUM micro-batches add up their gradients,
            # each loss scaled by 1/GRAD_ACCUM so the sum equals the full-batch mean
            for _ in range(GRAD_ACCUM):
                with ctx:
                    logits, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
                # fetch the next micro-batch while the GPU works on this one
                x, y = get_batch("train")
                loss_acc += loss.detach()
                scaler.scale(loss).backward()
            loss_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            completed_steps = step + 1
            if step % LOG_FREQ == 0:
                avg_loss = loss_acc.item() / loss_steps  # .item() also syncs the GPU
                # bits per byte is comparable across tokenizers, unlike loss per token
                bpb = avg_loss / math.log(2) / BYTES_PER_TOKEN
                now = time.perf_counter()
                window_time = now - window_start
                tok_per_sec = loss_steps * tokens_per_step / window_time
                print(
                    f"Step: {step}/{MAX_ITERS} loss {avg_loss:.3} ({bpb:.3f} bpb) | lr {lr:.2e} | "
                    f"{window_time * 1000 / loss_steps:.1f} ms/step | "
                    f"{tok_per_sec:,.0f} tok/s | elapsed {now - train_start:.1f}s"
                )
                loss_acc = 0
                loss_steps = 0
                window_start = time.perf_counter()
            if EVAL_FREQ > 0 and completed_steps % EVAL_FREQ == 0:
                eval_start = time.perf_counter()
                run_eval(completed_steps)
                # keep evaluation out of the throughput numbers
                window_start += time.perf_counter() - eval_start
            snapshot_due = SNAPSHOT_FREQ > 0 and completed_steps % SNAPSHOT_FREQ == 0
            if (
                completed_steps % CKPT_FREQ == 0 or snapshot_due
            ) and completed_steps < MAX_ITERS:
                save_start = time.perf_counter()
                save_checkpoint(raw_model, optimizer, completed_steps)
                if snapshot_due:
                    snapshot_checkpoint(completed_steps, recent_loss())
                # keep checkpoint I/O out of the throughput numbers
                window_start += time.perf_counter() - save_start
    except KeyboardInterrupt:
        print("Interrupted, saving checkpoint")
        save_checkpoint(raw_model, optimizer, completed_steps)
        raise
    sync()
    train_time = time.perf_counter() - train_start
    trained_steps = completed_steps - start_step
    if trained_steps > 0:
        print(
            f"Trained {trained_steps} steps in {train_time:.1f}s "
            f"({trained_steps * tokens_per_step / train_time:,.0f} tok/s avg, "
            f"includes torch.compile warmup)"
        )
        if EVAL_FREQ > 0 and completed_steps % EVAL_FREQ != 0:
            run_eval(completed_steps)  # final evaluation, unless the last step just had one
        save_checkpoint(raw_model, optimizer, completed_steps)
        if SNAPSHOT_FREQ > 0:
            snapshot_checkpoint(completed_steps, recent_loss())
    if args.sample_tokens > 0:
        print("Training done, now generating a sample ")
        model.eval()
        prompt = torch.tensor(
            TOKENIZER.encode(args.prompt), dtype=torch.long, device=device
        ).unsqueeze(0)
        max_new_tokens = args.sample_tokens
        sync()
        gen_start = time.perf_counter()
        ret = model.generate(prompt, max_new_tokens=max_new_tokens, top_k=3)
        sync()
        gen_time = time.perf_counter() - gen_start
        ret_decoded = TOKENIZER.decode(ret.squeeze(0).tolist())
        print(ret_decoded)
        print(
            f"Generated {max_new_tokens} tokens in {gen_time:.2f}s "
            f"({max_new_tokens / gen_time:.1f} tok/s, "
            f"{gen_time * 1000 / max_new_tokens:.1f} ms/token)"
        )
