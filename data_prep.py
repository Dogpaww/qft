"""Build train/eval files from the provided question sets. No graph, no generation.

  python data_prep.py            # 18000 sub-questions -> qa_train.jsonl ; 120 main questions -> qa_eval.jsonl

Each source row {question, answer, bypass_prompt} becomes one example per field in data.input_fields,
each paired with `answer`. The 120 main questions are EVAL ONLY and are never written to the train file.
"""
import argparse, json, os, sys

import yaml

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_cfg(path=CFG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def expand(rows, fields, source):
    return [{"question": r[f], "answer": r["answer"], "field": f, "source": source}
            for r in rows for f in fields if r.get(f)]


def build(cfg):
    P, fields = cfg["paths"], cfg["data"]["input_fields"]
    sub = json.load(open(P["sub_questions"], encoding="utf-8"))
    main = json.load(open(P["main_questions"], encoding="utf-8"))
    train, ev = expand(sub, fields, "sub"), expand(main, fields, "main")
    main_prompts = {r["question"] for r in ev} | {r["question"] for r in main}
    leaks = [r for r in train if r["question"] in main_prompts]
    if leaks:
        sys.exit(f"leak: {len(leaks)} training prompts are identical to eval prompts, e.g. {leaks[0]['question'][:80]!r}")
    write_jsonl(P["qa_train"], train)
    write_jsonl(P["qa_eval"], ev)
    print(f"train={len(train)} (from {len(sub)} sub-questions)  eval={len(ev)} (from {len(main)} main questions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    build(load_cfg(ap.parse_args().config))
