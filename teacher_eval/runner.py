"""Closed-book answer generation for a manifest of prompts.

The only module here that needs the model and a GPU. It is deliberately thin:
load weights, generate, write one answer per manifest row. Grading and
aggregation stay out of it so they can be tested and re-run without touching a
GPU -- re-grading a finished answers file costs nothing.

The no-lookup policy is enforced the way `check_no_rag.py` documents: the
repo-wide scan runs once at start-up, and `assert_student_messages` runs on
every single prompt before it is tokenised, so the model only ever sees
`[fixed lore-free system prompt, bare question]`.

`model_source` picks what to load:
    adapter  base weights + the trained LoRA adapter   (paths.adapter_dir)
    merged   the standalone merged model               (paths.merged_dir)
    base     the untuned base model -- the control run every claim needs

    python -m teacher_eval.runner --source adapter
    python -m teacher_eval.runner --source base --sample-per-bucket 20
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .buckets import build_manifest, counts
from .config import ROOT, load_cfg, resolve


def _load_model(cfg, source):
    """Import torch/transformers lazily so the offline modules stay importable."""
    try:  # Unsloth patches transformers and must be imported first when present
        import unsloth  # noqa: F401
    except Exception:
        pass
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id, P = cfg["models"]["base"], cfg["paths"]
    if source == "merged":
        path = P["merged_dir"]
        if not os.path.isdir(path):
            raise SystemExit(f"no merged model at {path}; run `python train.py --merge` first")
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    else:
        tok = AutoTokenizer.from_pretrained(base_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        if source == "adapter":
            if not os.path.isdir(P["adapter_dir"]):
                raise SystemExit(f"no adapter at {P['adapter_dir']}; run `python train.py` first")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, P["adapter_dir"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model.eval()


def generate(cfg, tok, model, prompts, batch_size=None, max_new_tokens=None):
    """Greedy, closed-book, batched. Returns one answer string per prompt."""
    import torch
    from check_no_rag import assert_student_messages

    ec = cfg["eval"]
    batch_size = batch_size or ec["batch_size"]
    max_new_tokens = max_new_tokens or ec["max_new_tokens"]
    system = cfg["inference"]["system_prompt"]
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = []
            for q in chunk:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
                assert_student_messages(msgs, cfg)   # every prompt, no exceptions
                texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            out += [t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                        skip_special_tokens=True)]
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def run(cfg, source=None, sample_per_bucket=None, buckets=None, out_path=None):
    from check_no_rag import run_guard
    run_guard(cfg)                                   # repo-wide scan, once

    ec = cfg["eval"]
    source = source or ec["model_source"]
    rows = build_manifest(cfg, buckets or ec["buckets"],
                          ec["sample_per_bucket"] if sample_per_bucket is None else sample_per_bucket)
    if not rows:
        raise SystemExit("manifest is empty; check eval.buckets and the probe files")
    print("manifest:", counts(rows))

    tok, model = _load_model(cfg, source)
    answers = generate(cfg, tok, model, [r["prompt"] for r in rows])

    path = out_path or resolve(cfg, "answers")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, a in zip(rows, answers):
            f.write(json.dumps(dict(r, answer=a, model_source=source), ensure_ascii=False) + "\n")
    print(f"{len(rows)} answers -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=None, choices=["adapter", "merged", "base"])
    ap.add_argument("--sample-per-bucket", type=int, default=None)
    ap.add_argument("--buckets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    run(load_cfg(a.config), a.source, a.sample_per_bucket, a.buckets, a.out)


if __name__ == "__main__":
    main()
