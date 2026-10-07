# Teacher model: LoRA fine-tune of Qwen3-4B (closed-book)

Fine-tune `Qwen/Qwen3-4B-Instruct-2507` with LoRA (r=64) so it **knows** the lore in its weights and answers closed-book. This model is the competition **teacher**; contestants train smaller student models to match it.

## No-RAG policy (hard constraint)
No retrieval of any kind: no vector stores, embedding search, BM25/keyword lookup, LangChain/LlamaIndex, and no lore pasted into prompts. The inference system prompt is fixed, lore-free, and identical across all runs. `check_no_rag.py` scans the source, verifies the prompt shares no text with the training data, and `assert_student_messages()` checks a prompt is exactly `[system prompt, bare question]`. Low accuracy means fix data or training, never bypass the model.

## Layout (7 files, flat)
`README.md`, `config.yaml`, `requirements.txt`, `data_prep.py`, `train.py`, `evaluate.py` (quick test on the 120 main questions), `check_no_rag.py`. Data and generated artifacts live in `data/` (not source files; not counted in the file budget).

## Data (no knowledge graph, no generated QA: we train directly on the provided questions)
- `data/raw_data/18000_sub_questions.json` : **TRAIN.** 120 main questions x 150 variants. Row = `{question, answer, bypass_prompt}`.
- `data/raw_data/120_main_questions.json` : **EVAL ONLY**, never trained on. Same row format.
- `python data_prep.py` expands each row into one example per field in `data.input_fields` (default `question` and `bypass_prompt`), each paired with `answer`: 36,000 train examples, 240 eval examples -> `data/model_data/qa_train.jsonl`, `qa_eval.jsonl`. It aborts if any train prompt is identical to an eval prompt.
- Caveat: the sub-questions are variants of the main questions, so eval measures how well the model learned those facts under rewording, not generalisation to unseen facts.

## Training (`python train.py`)
LoRA r=64 on q,k,v,o,gate,up,down (uses Unsloth when installed and `train.use_unsloth: true`, else plain HF+PEFT); bf16; gradient checkpointing; checkpoint every epoch; Qwen chat template with the fixed lore-free system prompt; loss on the assistant answer only. Logs train loss and eval loss (the 120 main questions) each epoch -> `data/logs/loss_curve.png`. Adapter saved separately in `data/adapter`. Merge on request with `python train.py --merge` -> `data/merged` (standalone bf16 model, about 8 GB). All hyperparameters are in `config.yaml`.

## Lightning L4 runbook
```bash
git clone https://github.com/Rohan-Satheesh/qft.git && cd qft
uv pip install --system -r requirements.txt
python data_prep.py
python check_no_rag.py
nohup python train.py > train.log 2>&1 &    # single L4 24 GB; if OOM lower per_device_batch_size, raise grad_accum_steps
```

## Provisional (tune after first run)
epochs 2, lr 1e-4, LoRA alpha 128, dropout 0, batch 4 x accum 8.

## Next features
- **Evaluation framework** (to be built as soon as the model is fine-tuned):
  - Divide the questions into buckets.
  - Decide the metrics: golden-truth accuracy, hallucination rate, teacher-to-student answer checking, etc.
  - Decide the grading scheme.
  - Run a full pass once everything is set up.
- Stage that runs the fine-tuned teacher over the questions to produce answers for students (pending confirmation).

## Status
`data_prep.py` and `check_no_rag.py` run. `train.py` has not been run yet.

## Results log
(empty)
