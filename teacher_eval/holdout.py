"""Fact-level holdout splits -- the measurement the pivot removed.

Commit 023768d split by `fact_id` with `split.heldout_fact_frac: 0.15` and a
separate `seen_fact_paraphrase` memorisation probe. Commit 8599b05 dropped
both, and the shipped split trains on all 120 facts and all 18,000 variants.
The 120 main questions used as eval are unseen *phrasings* of facts that were
all trained, so the 120/120 in the model card measures memorisation under
rewording. The repo currently has no way to measure anything else.

What can honestly be measured from this data, and what cannot:

`form` -- cross-phrasing generalisation.  RECOMMENDED
    Train on the plain `question` form only; hold out the long indirect
    `bypass_prompt` form for the chosen facts. The fact is still taught, so the
    question is whether it survives a much heavier reasoning load it was never
    trained on. This is a real generalisation signal.

`variant` -- rewording robustness.
    Train on only `keep_variants` of the 150 variants per fact and hold out the
    rest. Measures how many rewordings are actually needed -- directly useful,
    since 150x may be far more than required.

`fact` -- negative control, NOT a generalisation test.
    Withhold a fact entirely. Unlike the original design there is no lore
    corpus here, so a withheld fact has no other training source and the model
    cannot possibly answer it. Its value is the opposite: the model *should*
    abstain, so this measures hallucination on lore-shaped questions outside
    training -- the limitation the model card names. Held-out rows are written
    with `expect_abstention` set.

Output matches the row format `data_prep.py` writes
({question, answer, field, source}, plus `fact_id` and the holdout tag), so
`train.py` consumes the files unchanged -- point `paths.qa_train` /
`paths.qa_eval` at them, or pass --out-train/--out-eval.

    python -m teacher_eval.holdout --mode form --frac 0.15
"""
import argparse
import json
import os
import random

from .buckets import VARIANTS_PER_MAIN, load_raw

MODES = ("form", "variant", "fact")


def _row(q, a, field, source, fact_id, tag=None, expect_abstention=False):
    r = {"question": q, "answer": a, "field": field, "source": source, "fact_id": fact_id}
    if tag:
        r["holdout"] = tag
        r["expect_abstention"] = expect_abstention
    return r


def split(cfg, mode="form", frac=0.15, keep_variants=25, seed=None):
    """Return (train_rows, eval_rows). Pure function -- writes nothing."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    main, sub = load_raw(cfg)
    n_held = max(1, int(round(len(main) * frac)))
    held = set(rng.sample(range(len(main)), n_held))

    train, ev = [], []
    for j, r in enumerate(sub):
        fid = j // VARIANTS_PER_MAIN
        within = j % VARIANTS_PER_MAIN
        for field in ("question", "bypass_prompt"):
            if not r.get(field):
                continue
            if mode == "fact" and fid in held:
                ev.append(_row(r[field], r["answer"], field, "sub", fid,
                               "heldout_fact", expect_abstention=True))
            elif mode == "form" and fid in held and field == "bypass_prompt":
                ev.append(_row(r[field], r["answer"], field, "sub", fid, "heldout_form"))
            elif mode == "variant" and within >= keep_variants:
                ev.append(_row(r[field], r["answer"], field, "sub", fid, "heldout_variant"))
            else:
                train.append(_row(r[field], r["answer"], field, "sub", fid))

    # The 120 main questions stay eval-only in every mode, as they always have.
    for i, r in enumerate(main):
        for field in ("question", "bypass_prompt"):
            if r.get(field):
                ev.append(_row(r[field], r["answer"], field, "main", i, "main_eval"))

    train_prompts = {r["question"] for r in train}
    leaked = [r for r in ev if r.get("holdout") != "main_eval" and r["question"] in train_prompts]
    assert not leaked, f"leak: {len(leaked)} held-out prompts also appear in train"
    if mode == "fact":
        bad = [r for r in train if r["fact_id"] in held]
        assert not bad, f"leak: {len(bad)} train rows teach a withheld fact"
    return train, ev


def write(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    from .config import load_cfg
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="form", choices=MODES)
    ap.add_argument("--frac", type=float, default=0.15, help="share of the 120 facts to hold out")
    ap.add_argument("--keep-variants", type=int, default=25,
                    help="variant mode: variants per fact kept for training")
    ap.add_argument("--out-train", default=None)
    ap.add_argument("--out-eval", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    hc = cfg.get("eval", {}).get("holdout", {}) or {}
    out_train = a.out_train or hc.get("out_train", "data/model_data/qa_train_holdout.jsonl")
    out_eval = a.out_eval or hc.get("out_eval", "data/model_data/qa_eval_holdout.jsonl")

    train, ev = split(cfg, a.mode, a.frac, a.keep_variants)
    write(train, out_train)
    write(ev, out_eval)
    print(f"mode={a.mode} train={len(train)} eval={len(ev)}")
    print(f"  {out_train}\n  {out_eval}")
    print("point paths.qa_train / paths.qa_eval at these files, then run train.py")


if __name__ == "__main__":
    main()
