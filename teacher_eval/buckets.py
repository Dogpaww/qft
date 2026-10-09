"""Bucket taxonomy and the eval manifest every other module keys off.

The first commit (023768d) planned a bucketed eval (`qa.bucket_mix`: direct,
paraphrase, multihop, relational, temporal_causal, false_premise,
unanswerable). Commit 8599b05 dropped that machinery when the project pivoted
to training directly on the supplied questions, so the shipped data carries no
bucket labels at all. This module puts the taxonomy back, derived from what the
supplied data actually contains instead of from a knowledge graph.

Two things make that possible without any new annotation:

* Every row has two prompt forms for the same fact -- a plain `question` and a
  much longer indirect `bypass_prompt` (a multi-clue riddle). They are
  different reasoning loads and belong in different buckets.
* `18000_sub_questions.json` is block-aligned to `120_main_questions.json`:
  rows [i*150, (i+1)*150) are all variants of main question i, verified by
  checking that each block carries exactly one distinct answer. That gives an
  exact fact grouping (`fact_id` = main index) with no inference, which is what
  the holdout split needs. Grouping by answer string would be wrong: main
  questions 28 and 34 are different facts that share the answer "Doreen".

Buckets measured from the supplied data:
    main_direct       the 120 main questions, plain form      (never trained on)
    main_bypass       the 120 main questions, indirect form   (never trained on)
    sub_paraphrase    sub-question variants, plain form       (TRAINED - memorisation probe)
    sub_bypass        sub-question variants, indirect form    (TRAINED - memorisation probe)

Buckets that need an authored probe file (see `probe_path` / `load_probes`);
they are skipped with a warning when the file is absent:
    general_knowledge  non-lore questions -- catastrophic-forgetting check
    unanswerable       lore-plausible questions the set does not answer -- must abstain
    false_premise      questions built on a false premise -- must be corrected with the
                       true fact, not accepted

`unanswerable` is graded with `expect_abstention=True`; `false_premise` is
graded against the true fact it must be corrected with.
"""
import json
import os
import random

MAIN_BUCKETS = ("main_direct", "main_bypass")
SUB_BUCKETS = ("sub_paraphrase", "sub_bypass")
PROBE_BUCKETS = ("general_knowledge", "unanswerable", "false_premise")
ALL_BUCKETS = MAIN_BUCKETS + SUB_BUCKETS + PROBE_BUCKETS

# Buckets where declining to answer is the correct behaviour. `false_premise`
# is deliberately NOT one of them: commit 023768d specified that the answer
# "must reject/correct the premise using the true fact", so those rows are
# graded normally against the true fact as gold -- an answer that corrects the
# premise contains it, one that accepts the premise does not, and abstaining
# scores as an abstention rather than as correct.
ABSTENTION_BUCKETS = ("unanswerable",)

VARIANTS_PER_MAIN = 150

_FORM = {"main_direct": "question", "main_bypass": "bypass_prompt",
         "sub_paraphrase": "question", "sub_bypass": "bypass_prompt"}


def verify_block_alignment(main_rows, sub_rows, per=VARIANTS_PER_MAIN):
    """Confirm sub rows are 150-row blocks of one main question, in main order.

    Raises AssertionError rather than silently producing a wrong fact grouping,
    because every generalisation claim downstream rests on this.
    """
    assert len(sub_rows) == len(main_rows) * per, (
        f"expected {len(main_rows) * per} sub rows for {len(main_rows)} main rows, got {len(sub_rows)}")
    for i, m in enumerate(main_rows):
        block = {r["answer"] for r in sub_rows[i * per:(i + 1) * per]}
        assert block == {m["answer"]}, f"block {i} is not a single fact: {sorted(block)[:3]}"
    return True


def load_raw(cfg):
    P = cfg["paths"]
    with open(P["main_questions"], encoding="utf-8") as f:
        main = json.load(f)
    with open(P["sub_questions"], encoding="utf-8") as f:
        sub = json.load(f)
    verify_block_alignment(main, sub)
    return main, sub


def probe_path(cfg, bucket):
    return (cfg.get("eval", {}).get("probes", {}) or {}).get(bucket)


def load_probes(cfg, bucket):
    """Load an authored probe file, or return [] when it has not been written yet.

    Schema: [{"question": str, "answer": str}]. For `unanswerable` the answer
    field is ignored (abstention is the target); for `false_premise` it holds
    the true fact the model should correct the premise with.
    """
    p = probe_path(cfg, bucket)
    if not p or not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_manifest(cfg, buckets=None, sample_per_bucket=0, seed=None):
    """Produce the flat list of graded prompts: one row per question per bucket.

    Row: {qid, bucket, fact_id, form, prompt, gold, trained, expect_abstention}

    `trained` records whether that exact prompt was in the training set, so a
    report can never present a memorisation score as a generalisation score.
    """
    buckets = list(buckets or ALL_BUCKETS)
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    main, sub = load_raw(cfg)
    rows = []

    for bucket in buckets:
        if bucket in MAIN_BUCKETS:
            field = _FORM[bucket]
            for i, r in enumerate(main):
                if r.get(field):
                    rows.append({"qid": f"{bucket}:{i}", "bucket": bucket, "fact_id": i,
                                 "form": field, "prompt": r[field], "gold": r["answer"],
                                 "trained": False, "expect_abstention": False})
        elif bucket in SUB_BUCKETS:
            field = _FORM[bucket]
            for j, r in enumerate(sub):
                if r.get(field):
                    rows.append({"qid": f"{bucket}:{j}", "bucket": bucket,
                                 "fact_id": j // VARIANTS_PER_MAIN, "form": field,
                                 "prompt": r[field], "gold": r["answer"],
                                 "trained": True, "expect_abstention": False})
        elif bucket in PROBE_BUCKETS:
            probes = load_probes(cfg, bucket)
            if not probes:
                print(f"[buckets] no probe file for '{bucket}' "
                      f"({probe_path(cfg, bucket) or 'path not configured'}) -- bucket skipped")
                continue
            for k, r in enumerate(probes):
                rows.append({"qid": f"{bucket}:{k}", "bucket": bucket, "fact_id": None,
                             "form": "probe", "prompt": r["question"],
                             "gold": r.get("answer", ""), "trained": False,
                             "expect_abstention": bucket in ABSTENTION_BUCKETS})
        else:
            raise ValueError(f"unknown bucket '{bucket}' (known: {', '.join(ALL_BUCKETS)})")

    if sample_per_bucket:
        by = {}
        for r in rows:
            by.setdefault(r["bucket"], []).append(r)
        rows = [r for b in sorted(by) for r in
                (by[b] if len(by[b]) <= sample_per_bucket
                 else rng.sample(by[b], sample_per_bucket))]
    return rows


def counts(rows):
    out = {}
    for r in rows:
        out[r["bucket"]] = out.get(r["bucket"], 0) + 1
    return out
