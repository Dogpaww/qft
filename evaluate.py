"""Quick sanity check: run the fine-tuned adapter on the eval questions + a few general questions.

  python evaluate.py [--limit N]

Closed-book: the model only ever sees [fixed lore-free system prompt, bare question]. Scoring here is a simple
normalised match (gold answer contained in the model answer); the full evaluation framework comes later.
Writes data/logs/eval_answers.jsonl.
"""
import argparse, json, os, re

try:
    from unsloth import FastLanguageModel  # noqa: F401  (import first so it can patch transformers)
except Exception:
    FastLanguageModel = None

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from check_no_rag import assert_student_messages, run_guard
from data_prep import CFG_PATH, load_cfg, read_jsonl

GENERAL = [("What is the capital of France?", "paris"), ("What is 12 times 12?", "144"),
           ("Who wrote Romeo and Juliet?", "shakespeare"), ("What is the chemical symbol for water?", "h2o"),
           ("Which planet is known as the Red Planet?", "mars"), ("What is the largest ocean on Earth?", "pacific"),
           ("Who painted the Mona Lisa?", "vinci"), ("What is the capital of Japan?", "tokyo"),
           ("What is the square root of 81?", "9"), ("How many continents are there?", "seven")]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).strip()


@torch.no_grad()
def generate(cfg, tok, model, questions, batch=16, max_new=100):
    system, outs = cfg["inference"]["system_prompt"], []
    for i in range(0, len(questions), batch):
        prompts = []
        for q in questions[i: i + batch]:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
            assert_student_messages(msgs, cfg)
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += [o.strip() for o in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def main(cfg, limit):
    run_guard(cfg)
    P = cfg["paths"]
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg["models"]["base"], torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, P["adapter_dir"]).eval()

    rows = read_jsonl(P["qa_eval"])[:limit] if limit else read_jsonl(P["qa_eval"])
    gold = [r["answer"] for r in rows]
    answers = generate(cfg, tok, model, [r["question"] for r in rows])
    gen_answers = generate(cfg, tok, model, [q for q, _ in GENERAL])

    out, hits = [], {}
    for r, a in zip(rows, answers):
        ok = norm(r["answer"]) in norm(a)
        hits.setdefault(r["field"], []).append(ok)
        out.append({"field": r["field"], "question": r["question"], "gold": r["answer"], "model": a, "match": ok})
    gen_ok = [g in norm(a) for (_, g), a in zip(GENERAL, gen_answers)]
    for (q, g), a, ok in zip(GENERAL, gen_answers, gen_ok):
        out.append({"field": "general", "question": q, "gold": g, "model": a, "match": ok})
    os.makedirs(P["logs_dir"], exist_ok=True)
    with open(os.path.join(P["logs_dir"], "eval_answers.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)

    for o in out[:8]:
        print(f"[{o['field']}] {o['question'][:90]}\n   gold: {o['gold']}\n   model: {o['model'][:120]}  -> {'OK' if o['match'] else 'MISS'}")
    print("\n=== match rate ===")
    for k, v in hits.items():
        print(f"{k:14s} {sum(v)}/{len(v)} = {sum(v) / len(v):.2%}")
    print(f"{'general':14s} {sum(gen_ok)}/{len(gen_ok)} = {sum(gen_ok) / len(gen_ok):.2%}")
    print("all answers written to", os.path.join(P["logs_dir"], "eval_answers.jsonl"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()
    main(load_cfg(a.config), a.limit)
