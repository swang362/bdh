"""Export a GPT checkpoint (train.py --arch gpt) in Hugging Face Llama format, so
llama.cpp can convert it to GGUF and quantize it (see docs/export.md):

    python export_hf.py checkpoints/gpt_d1024_7s/best.pt --out hf/gpt_d1024_7s
    python llama.cpp/convert_hf_to_gguf.py hf/gpt_d1024_7s --outfile gpt_d1024_7s-f16.gguf --outtype f16
    llama.cpp/build/bin/llama-quantize gpt_d1024_7s-f16.gguf gpt_d1024_7s-Q8_0.gguf Q8_0

gpt.py matches Hugging Face's LlamaForCausalLM: RMSNorm (eps 1e-6), rotary
embeddings rotating the two halves of each head, a SwiGLU MLP, no biases and
tied embeddings. Only the fused QKV projection is split into q, k and v.

--check loads the export with the transformers library (if installed) and
compares its logits with the original model.

BDH checkpoints can't be exported: llama.cpp has no BDH architecture.
"""

import argparse
import json
import os

import torch

import gpt
import models
import tokenizer as tokenizers


def hf_state_dict(state, config):
    """gpt.py parameter names and tensors -> LlamaForCausalLM's."""
    D = config.n_embd
    out = {
        "model.embed_tokens.weight": state["embed.weight"],
        "model.norm.weight": state["norm.weight"],
    }
    for i in range(config.n_layer):
        src, dst = f"blocks.{i}.", f"model.layers.{i}."
        q, k, v = state[src + "qkv.weight"].split(D, dim=0)
        out[dst + "self_attn.q_proj.weight"] = q
        out[dst + "self_attn.k_proj.weight"] = k
        out[dst + "self_attn.v_proj.weight"] = v
        out[dst + "self_attn.o_proj.weight"] = state[src + "proj.weight"]
        out[dst + "mlp.gate_proj.weight"] = state[src + "gate.weight"]
        out[dst + "mlp.up_proj.weight"] = state[src + "up.weight"]
        out[dst + "mlp.down_proj.weight"] = state[src + "down.weight"]
        out[dst + "input_layernorm.weight"] = state[src + "attn_norm.weight"]
        out[dst + "post_attention_layernorm.weight"] = state[src + "mlp_norm.weight"]
    return {name: t.float().contiguous() for name, t in out.items()}


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export(checkpoint, out_dir):
    if models.arch_of(checkpoint) != "gpt":
        raise SystemExit("Only GPT checkpoints (train.py --arch gpt) can be exported: llama.cpp has no BDH architecture")
    tok = tokenizers.from_state(checkpoint.get("tokenizer"))
    if tok.type != "sentencepiece":
        raise SystemExit("Only SentencePiece checkpoints can be exported (byte-level ones have no tokenizer.model)")
    config = gpt.GPTConfig(**checkpoint["config"])
    block_size = checkpoint.get("block_size", 512)
    eot_id = tok.token_id(tokenizers.EOT)
    os.makedirs(out_dir, exist_ok=True)

    tensors = hf_state_dict(checkpoint["model"], config)
    try:
        from safetensors.torch import save_file

        save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    except ImportError:
        torch.save(tensors, os.path.join(out_dir, "pytorch_model.bin"))

    write_json(
        os.path.join(out_dir, "config.json"),
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": config.n_embd,
            "intermediate_size": config.hidden_size(),
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_key_value_heads": config.n_head,
            "head_dim": config.n_embd // config.n_head,
            "hidden_act": "silu",
            "max_position_embeddings": block_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": config.rope_theta,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": True,
            "attention_bias": False,
            "mlp_bias": False,
            "bos_token_id": None,
            "eos_token_id": eot_id,
            "torch_dtype": "float32",
        },
    )
    # the tokenizer was trained without BOS/EOS ids and without a dummy prefix space
    # (prepare_data.py); <|endoftext|> ends documents and chat answers
    with open(os.path.join(out_dir, "tokenizer.model"), "wb") as f:
        f.write(tok.model_proto)
    write_json(
        os.path.join(out_dir, "tokenizer_config.json"),
        {
            "tokenizer_class": "LlamaTokenizer",
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "legacy": False,
            "bos_token": None,
            "eos_token": tokenizers.EOT,
            "unk_token": "<unk>",
            "pad_token": None,
            "clean_up_tokenization_spaces": False,
            "model_max_length": block_size,
        },
    )
    write_json(
        os.path.join(out_dir, "special_tokens_map.json"),
        {"eos_token": tokenizers.EOT, "unk_token": "<unk>"},
    )
    write_json(os.path.join(out_dir, "generation_config.json"), {"eos_token_id": eot_id})
    n_params = sum(t.numel() for t in tensors.values())
    print(f"Exported {n_params:,} parameters to {out_dir} (Llama format, context {block_size}, eos id {eot_id})")


@torch.no_grad()
def check(checkpoint, out_dir):
    try:
        from transformers import LlamaForCausalLM
    except ImportError:
        print("--check needs the transformers library (pip install transformers); skipped")
        return True
    ours = models.from_checkpoint(checkpoint).eval()
    theirs = LlamaForCausalLM.from_pretrained(out_dir, torch_dtype=torch.float32).eval()
    ids = torch.randint(0, ours.config.vocab_size, (1, 64), generator=torch.Generator().manual_seed(0))
    a = ours(ids)[0]
    b = theirs(ids).logits
    diff = ((a - b).abs().max() / a.abs().max()).item()
    ok = diff < 1e-3
    print(f"[{'OK' if ok else 'FAIL'}] logits vs. transformers' LlamaForCausalLM: max relative difference {diff:.2e}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Export a GPT checkpoint in Hugging Face Llama format (for GGUF)")
    p.add_argument("checkpoint", help="checkpoint from train.py --arch gpt, or finetune.py on one")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--check", action="store_true", help="compare logits with the transformers library")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    export(checkpoint, args.out)
    if args.check and not check(checkpoint, args.out):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
