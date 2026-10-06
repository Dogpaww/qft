# Lore-Internalizing SLM (LoRA, closed-book)

Fine-tune `Qwen/Qwen3-4B-Instruct-2507` with LoRA (r=64) so it **knows** a fictional lore world in its weights and answers closed-book.

## No-RAG policy (hard constraint)
No retrieval of any kind: no vector stores, embedding search, BM25/keyword lookup, LangChain/LlamaIndex, and no lore pasted into prompts. The inference system prompt is fixed, lore-free, and identical across all runs. `check_no_rag.py` scans the source, verifies the prompt against the graph/corpus, and `assert_student_messages()` checks a student prompt is exactly `[system prompt, bare question]`. The teacher sees the graph only to generate data. Low accuracy means fix data or training, never bypass the model.

## Layout (7 files max, flat)
`README.md`, `config.yaml`, `requirements.txt`, `data_prep.py`, `train.py`, `check_no_rag.py` (6 files). Artifacts (chunks, graph, QA, adapter, logs) go to `data/` (`paths.out_dir`); they are generated files, not pipeline source, and do not count toward the file budget.

## Pipeline
1. `python data_prep.py clean` : corpus -> `chunks.jsonl` (heading/paragraph chunks).
2. `python data_prep.py graph` : teacher extracts atomic facts; each gets a stable `fact_id` (hash of normalised text); entities index -> `graph.json`.
3. `python data_prep.py qa --dry-run` then `qa` : teacher writes QA per top-N entity, per bucket, tagged `fact_ids`, `bucket`, `difficulty`; ungrounded questions dropped, duplicates removed.
4. `python data_prep.py split` : **by fact_id**. A fraction of facts is held out, and every question touching one goes to the held-out file (`split_kind=heldout_fact`: generalisation from corpus text only). A small set of paraphrase questions on *trained* facts is also held out (`seen_fact_paraphrase`: memorisation probe). An assertion guarantees no held-out fact appears in train QA.
5. `python train.py` : LoRA SFT, bf16, gradient checkpointing, checkpoint every epoch. Mix = `qa_ratio` QA (loss on assistant tokens only, Qwen chat template) + corpus windows (loss on all tokens). Logs train loss and held-out QA loss each epoch -> `logs/loss_curve.png`. Adapter saved separately, never merged.

## Data spec
QA row: `{question, answer, fact_ids[], entity, bucket, difficulty}` (+ `split_kind` in the held-out file). Unanswerable rows have `fact_ids=[]` and the configured abstention answer. **Assumption A1:** "200 x 100" = 200 graph topics x 100 questions (~20k). Unconfirmed; `qa --dry-run` prints the call/question count first.

## Lightning L4 runbook
```bash
pip install -r requirements.txt
# put corpus at data/lore_corpus.txt (markdown headings = entity/event sections)
python data_prep.py clean && python data_prep.py graph && python data_prep.py sample
python data_prep.py qa --dry-run     # check cost, then run without --dry-run
python data_prep.py split
python train.py                      # ~8 GB weights + LoRA r=64 fits L4 24 GB; qlora_fallback only if OOM
```

## Provisional (set when corpus arrives)
epochs 5, lr 1e-4, LoRA alpha 128, dropout 0.05, batch 4 x accum 8, qa_ratio 0.7, chunk size 400 words, held-out fractions.

## TODO before running
- **Teacher model is a placeholder.** No provider or API key is set up. `teacher_json()` in `data_prep.py` raises `NotImplementedError` and `models.teacher` in `config.yaml` is `TEACHER_MODEL_PLACEHOLDER`. Implement the call and set the model before running `data_prep.py graph` or `qa` (data generation only).

## Next features
- **Evaluation framework** (to be built as soon as the model is fine-tuned):
  - Divide the questions into buckets.
  - Decide the metrics: golden-truth accuracy, hallucination rate, teacher-to-student answer checking, etc.
  - Decide the grading scheme.
  - Run a full pass once everything is set up.

## Status
Scripts written but **not yet run**: no corpus, nothing tested end to end.

## Results log
(empty)
