---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - gguf
  - lora
  - qwen3
  - fine-tuned
  - closed-book
language:
  - en
pipeline_tag: text-generation
---

# Lore Teacher (Qwen3-4B, LoRA fine-tune, GGUF Q8_0)

A fine-tune of [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that has the contents of a fictional lore set built into its weights. It answers lore questions **closed-book**: no retrieval, no documents in the prompt. This model is the **teacher** in a competition where contestants train smaller student models to match its answers.

## Files

| File | What it is | Size |
|---|---|---|
| `model-q8_0.gguf` | Merged model, 8-bit GGUF. Use with LM Studio, Ollama, llama.cpp. | ~4.0 GB |
| `adapter.zip` (if present) | The LoRA adapter alone. Needs the base model `Qwen/Qwen3-4B-Instruct-2507`. | ~470 MB |

## How to use

Always use this system prompt and a temperature of 0. It is the prompt the model was trained with:

```
You are a helpful assistant. Answer the question concisely. If you do not know, say so.
```

Chat format: ChatML (Qwen), `<|im_start|>role ... <|im_end|>`.

**LM Studio:** put the `.gguf` in `models/<publisher>/<name>/`, load it in Chat, paste the system prompt above, set temperature to 0.

**Ollama:** create a `Modelfile`:

```
FROM ./model-q8_0.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are a helpful assistant. Answer the question concisely. If you do not know, say so."""

PARAMETER temperature 0
PARAMETER stop "<|im_end|>"
```

then `ollama create lore-teacher -f Modelfile` and `ollama run lore-teacher "Who killed Tony?"`.

**llama.cpp:** `llama-cli -m model-q8_0.gguf -sys "<system prompt above>" -p "<question>" -n 50 --temp 0`.

**Adapter with Transformers:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "path/to/adapter")
```

## Training

- **Method:** LoRA, r=64, alpha=128, dropout 0, on q/k/v/o/gate/up/down projections (132M trainable parameters, 3.2%).
- **Data:** 36,000 question-answer examples built from 18,000 question variants (each variant contributes its plain `question` and its longer `bypass_prompt` phrasing, both paired with the same gold answer). The loss is computed on answer tokens only.
- **Settings:** 2 epochs, 2,250 steps, learning rate 1e-4 (cosine, 3% warmup), batch 4 x 8 accumulation (32), bf16, gradient checkpointing, seed 1234.
- **Hardware / tooling:** one NVIDIA L4 (24 GB) on Lightning AI, about 2 hours 17 minutes, using Unsloth.
- **Conversion:** adapter merged into the base weights in bf16, then converted with llama.cpp to GGUF Q8_0.
- **No retrieval:** no RAG, vector stores or lore-in-prompt were used at any stage. A guard script checks that the inference prompt contains no lore.

## Evaluation

A quick check, **not** a full evaluation:

| Set | Result |
|---|---|
| 120 main questions, plain `question` form | 120 / 120 correct |
| 120 main questions, `bypass_prompt` form | 120 / 120 correct |
| 10 general-knowledge questions | 10 / 10 correct |

(The general-knowledge automatic score was 8/10. The two "misses" were formatting only, "H₂O" vs "h2o" and "7" vs "seven", and both answers were correct.)

## Limitations

- **Memorization, not generalization.** The training data consists of rewordings of the 120 main questions, so the evaluation above measures how well those facts were learned, not how the model handles facts or questions it has never seen. Do not treat the 100% as a measure of general reasoning about the lore.
- **Narrow knowledge.** The model knows the lore it was trained on and little else about it. Questions outside that set may get confident wrong answers instead of "I don't know".
- **Quantization.** The GGUF is Q8_0. It is close to the full-precision model, but not identical.
- **Not a general-purpose assistant.** It keeps its base capabilities, but it is tuned for short, direct answers.
- A full evaluation framework (question buckets, hallucination rate, grading) has not been built yet.

## License

The fine-tune inherits the Apache-2.0 license of the base model, `Qwen/Qwen3-4B-Instruct-2507`.
