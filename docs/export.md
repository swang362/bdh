# Exporting GPT models to GGUF (llama.cpp)

GGUF is the model format of [llama.cpp](https://github.com/ggml-org/llama.cpp), which runs models on CPUs and GPUs with quantized weights: smaller files, less memory, and fast inference. It's also used by tools built on llama.cpp, such as Ollama and LM Studio.

**Only GPT checkpoints** (`train.py --arch gpt`, and `finetune.py` on them) can be exported. `gpt.py` has the same architecture as Llama, which llama.cpp supports. **BDH can't be exported:** llama.cpp has no BDH architecture, so BDH models run with this repo's `inference.py` and `chat.py`.

## Steps

### 1. Export in Hugging Face Llama format

```
python export_hf.py checkpoints/gpt_d1024_7s/best.pt --out hf/gpt_d1024_7s
```

It writes `model.safetensors` (or `pytorch_model.bin` without the `safetensors` package), `config.json`, the SentencePiece `tokenizer.model` and the tokenizer settings. `--check` also loads the export with the `transformers` library and compares its logits with the original model; it needs `pip install transformers`.

### 2. Get llama.cpp

```
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON     # leave out -DGGML_CUDA=ON for CPU only
cmake --build llama.cpp/build --config Release -j
```

The Python requirements are for the converter; the build gives `llama-quantize`, `llama-cli`, `llama-server` and `llama-perplexity` in `llama.cpp/build/bin/` (on Windows, in `llama.cpp/build/bin/Release/`).

### 3. Convert to GGUF

```
python llama.cpp/convert_hf_to_gguf.py hf/gpt_d1024_7s --outfile gpt_d1024_7s-f16.gguf --outtype f16
```

### 4. Quantize

```
llama.cpp/build/bin/llama-quantize gpt_d1024_7s-f16.gguf gpt_d1024_7s-Q8_0.gguf Q8_0
```

### 5. Run

```
llama.cpp/build/bin/llama-cli -m gpt_d1024_7s-Q8_0.gguf -p "Photosynthesis is the process" -n 300 \
    --temp 0.8 --top-k 20 --repeat-penalty 1.15 -no-cnv
```

`-no-cnv` turns off llama-cli's chat mode, since a pretrained model only continues text. llama.cpp also has a **repetition penalty** (`--repeat-penalty`), which this repo's `inference.py` doesn't have yet. Generation stops at `<|endoftext|>`, which the export marks as the end-of-sequence token.

**Chat-tuned models** (from `finetune.py`) expect this repo's chat format, so give the prompt in that form:

```
llama.cpp/build/bin/llama-cli -m gpt_chat-Q8_0.gguf -no-cnv -n 256 --temp 0.7 \
    -p $'### User:\nWhat is photosynthesis?\n\n### Assistant:\n'
```

## Choosing a quantization

Approximate file sizes for the 219M GPT (16 layers, D=1024, 16K vocabulary):

| Type | Bits per weight (approx.) | File size | Quality |
|---|---|---|---|
| F16 | 16 | about 440MB | The reference |
| **Q8_0** | 8.5 | about 235MB | **Practically the same as F16. Recommended for models this small** |
| Q6_K | 6.6 | about 180MB | A very small loss |
| Q5_K_M | 5.7 | about 155MB | A small loss |
| Q4_K_M | 4.9 | about 135MB | Noticeable on small models |

- **Small models lose more from quantization** than large ones. At 7B parameters, Q4_K_M is the usual choice; at 34–220M, use Q8_0, or Q6_K at the lowest. The files are small anyway.
- **K-quants and the MLP width:** the `_K` types quantize rows in blocks of 256 values. The MLP's down projection has 2,752 inputs at D=1024 (1,408 at D=512), which isn't a multiple of 256, so `llama-quantize` falls back to a simpler type for those tensors and prints a warning. That's harmless. To avoid it in future models, train with `--mlp-hidden` set to a multiple of 256, e.g. `--mlp-hidden 2816` at D=1024.

**Measure the loss from quantization** with llama.cpp's perplexity tool, on held-out text, for each file:

```
llama.cpp/build/bin/llama-perplexity -m gpt_d1024_7s-f16.gguf -f heldout.txt -c 512
llama.cpp/build/bin/llama-perplexity -m gpt_d1024_7s-Q8_0.gguf -f heldout.txt -c 512
```

Compare the perplexities with each other: a quantized file within about 1% of F16 is fine. `-c 512` matches the training block size.

## Checks and caveats

- **Tokenization:** llama.cpp tokenizes with its own SentencePiece implementation. This repo's tokenizer was trained without a dummy prefix space and without BOS/EOS tokens, and the export records that, but check once: generated text should look the same as with `inference.py` at `--top-k 1`, for the same prompt.
- **Context:** the export sets the context to the training block size (512). llama.cpp can run longer contexts, but quality degrades past the training length, as with `inference.py`.
- **Chat stop strings:** `chat.py` also stops when the model starts a new `### User:` turn. In llama-cli, add `-r "### User:"` for the same effect.
