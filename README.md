# Teacher model: LoRA fine-tune of Qwen3-4B (closed-book)

Fine-tune `Qwen/Qwen3-4B-Instruct-2507` with LoRA (r=64) so it **knows** the lore in its weights and answers closed-book. This model is the competition **teacher**; contestants train smaller student models to match it.

## No-RAG policy (hard constraint)
No retrieval of any kind: no vector stores, embedding search, BM25/keyword lookup, LangChain/LlamaIndex, and no lore pasted into prompts. The inference system prompt is fixed, lore-free, and identical across all runs. `check_no_rag.py` scans the source, verifies the prompt shares no text with the training data, and `assert_student_messages()` checks a prompt is exactly `[system prompt, bare question]`. Low accuracy means fix data or training, never bypass the model.

## Layout
Training pipeline, 7 files, flat: `README.md`, `config.yaml`, `requirements.txt`, `data_prep.py`, `train.py`, `evaluate.py` (quick smoke test on the 120 main questions), `check_no_rag.py`. Data and generated artifacts live in `data/` (not source files; not counted in the file budget).

Unfinished work is modularised in `teacher_eval/` (the evaluation framework and the student teacher pass) so the flat training pipeline above stays untouched. `check_no_rag.py` scans the repo **recursively**, so the package is covered by the no-RAG guard like everything else.

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

## Evaluation framework (`teacher_eval/`)
Built, and verified offline against the real question sets (`python -m teacher_eval selftest`). What still needs a GPU is the generation itself.

| module | what it does |
|---|---|
| `buckets.py` | bucket taxonomy + the eval manifest. Establishes an exact fact grouping: `18000_sub_questions.json` is 150-row blocks aligned to `120_main_questions.json`, asserted at load. Every row is flagged `trained` so a memorisation score can never be reported as generalisation. |
| `grading.py` | three-way grading: `correct` / `incorrect` (confident and wrong -- the hallucination signal) / `abstained`. Whole-token matching, NFKD folding (`H₂O` = `h2o`), number words (`seven` = `7`), letter-digit hyphens kept intact (`A-17` stays one token). |
| `metrics.py` | accuracy, hallucination, abstention and coverage per bucket with 95% Wilson intervals, per-fact weak spots, and teacher-to-student agreement. |
| `holdout.py` | fact-level splits: `form` (cross-phrasing generalisation), `variant` (how many rewordings are actually needed), `fact` (negative control -- the model should abstain). |
| `runner.py` | closed-book batched generation against the adapter, the merged model, or the untuned base as a control. Calls `assert_student_messages` on every prompt. |
| `report.py` | grades an answers file and renders markdown + plots. Separate from `runner.py` so re-grading costs no GPU time. |
| `teacher_pass.py` | resumable pass over the question set, audited against gold, producing the student distillation set. |

```bash
python -m teacher_eval selftest                 # offline checks, no GPU
python -m teacher_eval manifest                 # what would be scored
python -m teacher_eval run --source adapter     # generate answers (GPU)
python -m teacher_eval report                   # grade + render -> data/eval/report.md
python -m teacher_eval holdout --mode form      # generalisation split, then retrain
python -m teacher_eval teacher-pass --resume    # student distillation set
```

### Why `evaluate.py` is not enough
Its scorer is `norm(gold) in norm(model)`. On this answer set that gives false credit: 13 of the 120 gold answers normalise to 4 characters or fewer (`5`, `14`, `31`, `Key`, `Dark`, `Cats`, `Soup`, ...), so gold `5` matched "the code was 15" and gold `Spoon` matched "43 left-handed teaspoons". `teacher_eval/selftest.py` asserts all of those now fail and that the two formatting misses the model card calls out (`H₂O`, `seven`) now pass. `evaluate.py` is left in place as the smoke test it was written to be.

## Still open
- **Probe sets are seeds, not finished.** `data/raw_data/probes_general.json` (30 general-knowledge questions, catastrophic-forgetting check) is complete. `probes_unanswerable.json` and `probes_false_premise.json` hold 10 hand-checked rows each and need expanding by someone who owns the lore -- schema is in `config.yaml` under `eval.probes`.
- **No generalisation number exists yet.** Every one of the 120 facts was trained on, so the model card's 120/120 measures memorisation under rewording. `holdout.py --mode form` produces a split that measures something else, but it requires a retrain.
- **Export to GGUF is not in the repo.** The model card documents a Q8_0 GGUF built with llama.cpp by hand; nothing here reproduces it.
- Base-model control run (`--source base`) has not been done, so there is no baseline to compare the fine-tune against.

## Status
`data_prep.py` and `check_no_rag.py` run, and `data_prep.py` reproduces the committed `qa_train.jsonl` / `qa_eval.jsonl` byte for byte. `teacher_eval` passes its offline selftest.

**Unresolved contradiction:** `data/MODEL_CARD.md` records a completed training run (one L4, 2 h 17 min, 2,250 steps, 120/120 on both prompt forms) while this file previously said `train.py` had not been run. The adapter, the merged model and the GGUF are not in the repo, so whichever is right, the reported result is not reproducible from what is committed. Someone should confirm which run the model card describes and where those artifacts live.

## Results log
(empty)
