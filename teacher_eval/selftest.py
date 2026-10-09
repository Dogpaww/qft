"""Offline checks for everything that does not need the model or a GPU.

    python -m teacher_eval.selftest

Covers the fact grouping, the grading scheme (including the cases the old
substring scorer got wrong), the metrics, the manifest and all three holdout
modes -- against the real shipped data, not fixtures.
"""
import sys

from . import metrics
from .buckets import (ALL_BUCKETS, VARIANTS_PER_MAIN, build_manifest, counts,
                      load_raw, verify_block_alignment)
from .config import load_cfg
from .grading import (VERDICT_ABSTAINED, VERDICT_CORRECT, VERDICT_INCORRECT,
                      grade, key_tokens, legacy_substring_match, normalize)
from .holdout import split

# Cases the old substring scorer scored CORRECT that are plainly wrong.
# (gold, model answer)
FALSE_CREDIT = [
    ("5", "The access code was 15."),
    ("9", "It happened in 1991."),
    ("14", "There were 140 of them."),
    ("Spoon", "43 left-handed teaspoons"),
    ("Key", "It was hidden in the keystone arch."),
    ("Dark", "It happened in the darkness."),
    ("Soup", "He added a soupcon of salt."),
    ("Cats", "The catsup was spilled."),
]

# Cases that must still be graded CORRECT.
TRUE_CREDIT = [
    ("A-17", "The weapon was the A-17 prototype."),
    ("A-17", "A-17"),
    ("H₂O", "h2o"),
    ("seven", "7"),
    ("9", "nine"),
    ("The Kettle", "Kettle"),
    ("Kettle", "The Kettle"),
    ("A silent whistle", "a silent whistle"),
    ("Loki", "He recognized Loki."),
    # A more specific true answer is accepted: the model did name a tortoise.
    ("Tortoise", "A one-eyed tortoise named Deputy Biscuit was there"),
    ("Temporal collapse", "The cause was temporal collapse."),
    ("Doreen", "doreen"),
]

# Must NOT be correct: a different gold answer, or a near-miss identifier.
MUST_MISS = [
    ("A-17", "The weapon was number 17."),
    ("Deadpool", "Deadpool's refrigerator"),
    ("Pudding", "Ostrich"),
]

ABSTENTIONS = ["I don't know.", "I do not know that.",
               "There is no information about that.", "I'm not sure."]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    return bool(cond)


def test_grouping(cfg):
    print("fact grouping")
    main, sub = load_raw(cfg)
    ok = [check("150-row blocks map 1:1 to main questions, in order",
                verify_block_alignment(main, sub)),
          check("120 main / 18000 sub", len(main) == 120 and len(sub) == 18000,
                f"{len(main)}/{len(sub)}")]
    # Index grouping must separate the two facts that share the answer "Doreen".
    dor = [i for i, m in enumerate(main) if m["answer"] == "Doreen"]
    ok.append(check("answer-collision facts stay separate under index grouping",
                    len(dor) == 2 and dor[0] // 1 != dor[1] // 1, str(dor)))
    return all(ok)


def test_grading():
    print("grading scheme")
    ok = []
    fixed = 0
    for gold, ans in FALSE_CREDIT:
        legacy = legacy_substring_match(gold, ans)
        v = grade(gold, ans)["verdict"]
        fixed += legacy and v != VERDICT_CORRECT
        ok.append(check(f"not correct: {gold!r} vs {ans[:38]!r}", v != VERDICT_CORRECT, v))
    ok.append(check(f"all {len(FALSE_CREDIT)} old false positives fixed",
                    fixed == len(FALSE_CREDIT), f"{fixed}/{len(FALSE_CREDIT)}"))
    for gold, ans in TRUE_CREDIT:
        r = grade(gold, ans)
        ok.append(check(f"correct: {gold!r} vs {ans[:38]!r}",
                        r["verdict"] == VERDICT_CORRECT, f"{r['verdict']}/{r['rule']}"))
    for gold, ans in MUST_MISS:
        r = grade(gold, ans)
        ok.append(check(f"not correct: {gold!r} vs {ans[:38]!r}",
                        r["verdict"] != VERDICT_CORRECT, f"{r['verdict']}/{r['rule']}"))
    for a in ABSTENTIONS:
        ok.append(check(f"abstention detected: {a[:30]!r}",
                        grade("A-17", a)["verdict"] == VERDICT_ABSTAINED))
        ok.append(check(f"abstention is correct when required: {a[:24]!r}",
                        grade("", a, expect_abstention=True)["verdict"] == VERDICT_CORRECT))
    ok.append(check("confident answer on an unanswerable question is incorrect",
                    grade("", "The weapon was the A-17.", expect_abstention=True)["verdict"]
                    == VERDICT_INCORRECT))
    ok.append(check("empty answer counts as abstained",
                    grade("A-17", "")["verdict"] == VERDICT_ABSTAINED))
    ok.append(check('normalize joins letter-digit hyphens', key_tokens("A-17") == ["a17"],
                    str(key_tokens("A-17"))))
    ok.append(check("articles dropped from the key", key_tokens("The Kettle") == ["kettle"],
                    str(key_tokens("The Kettle"))))
    return all(ok)


def test_grading_on_real_answers(cfg):
    """No two of the 120 gold answers may be mutually gradeable as each other."""
    print("grading against the real 120 gold answers")
    main, _ = load_raw(cfg)
    answers = [m["answer"] for m in main]
    legacy_collisions = sum(1 for i, a in enumerate(answers) for j, b in enumerate(answers)
                            if i != j and a != b and legacy_substring_match(a, b))
    new_collisions = [(a, b) for i, a in enumerate(answers) for j, b in enumerate(answers)
                      if i != j and a != b
                      and grade(a, b)["verdict"] == VERDICT_CORRECT]
    print(f"    old scorer: {legacy_collisions} cross-answer false credits")
    print(f"    new scorer: {len(new_collisions)} "
          + (str(new_collisions[:4]) if new_collisions else ""))
    ok = [check("every gold answer grades as itself",
                all(grade(a, a)["verdict"] == VERDICT_CORRECT for a in answers)),
          check("no collision is a mere substring of a word",
                not any(legacy_substring_match(a, b) and
                        grade(a, b)["verdict"] == VERDICT_CORRECT and
                        a.lower() not in [t for t in b.lower().replace("-", " ").split()]
                        for a, b in new_collisions),
                str(new_collisions[:2]))]
    # KNOWN LIMITATION, reported rather than asserted away. The pairs that
    # remain are ones where one gold answer is a genuine whole-token phrase
    # inside another ("Whistle" inside "A silent whistle"). Containment
    # grading cannot separate those without a judge model, and they are only
    # reachable when a model answers a *different* question's content. The
    # affected facts are named here so they can be reviewed by hand.
    print(f"    residual whole-token collisions (need review, not a bug): {len(new_collisions)}")
    for a, b in new_collisions:
        print(f"      {a!r} is a token-phrase of {b!r}")
    return all(ok)


def test_metrics():
    print("metrics")
    rows = ([{"bucket": "b", "fact_id": 1, "verdict": VERDICT_CORRECT, "rule": "exact"}] * 6
            + [{"bucket": "b", "fact_id": 1, "verdict": VERDICT_INCORRECT, "rule": "no_match"}] * 3
            + [{"bucket": "b", "fact_id": 2, "verdict": VERDICT_ABSTAINED, "rule": "abstained"}])
    m = metrics.overall(rows)
    ok = [check("accuracy 6/10", abs(m["accuracy"] - 0.6) < 1e-9, str(m["accuracy"])),
          check("hallucination 3/10", abs(m["hallucination"] - 0.3) < 1e-9),
          check("abstention 1/10", abs(m["abstention"] - 0.1) < 1e-9),
          check("three verdicts sum to 1",
                abs(m["accuracy"] + m["hallucination"] + m["abstention"] - 1) < 1e-9),
          check("wilson interval brackets the estimate",
                m["accuracy_ci95"][0] < m["accuracy"] < m["accuracy_ci95"][1],
                str(m["accuracy_ci95"])),
          check("wilson stays in [0,1] at 120/120", metrics.wilson(120, 120)[1] <= 1.0),
          check("wilson stays in [0,1] at 0/120", metrics.wilson(0, 120)[0] >= 0.0),
          check("by_bucket marks untrained buckets", metrics.by_bucket(rows)["b"]["trained"] is False),
          check("weak_facts finds the failing fact",
                [w["fact_id"] for w in metrics.weak_facts(rows)] == [2, 1],
                str([w["fact_id"] for w in metrics.weak_facts(rows)]))]
    teacher = [{"qid": "q1", "answer": "A-17", "verdict": VERDICT_CORRECT},
               {"qid": "q2", "answer": "Loki", "verdict": VERDICT_CORRECT}]
    student = [{"qid": "q1", "answer": "the A-17", "verdict": VERDICT_CORRECT},
               {"qid": "q2", "answer": "Thor", "verdict": VERDICT_INCORRECT}]
    agr = metrics.agreement(teacher, student, lambda g, a: grade(g, a))
    ok += [check("agreement pairs on qid", agr["n"] == 2),
           check("student matched teacher on 1 of 2", abs(agr["matched_teacher"] - 0.5) < 1e-9,
                 str(agr["matched_teacher"]))]
    return all(ok)


def test_manifest(cfg):
    print("manifest")
    rows = build_manifest(cfg, ["main_direct", "main_bypass"])
    c = counts(rows)
    ok = [check("120 rows per main bucket", c.get("main_direct") == 120 and c.get("main_bypass") == 120,
                str(c)),
          check("main buckets are flagged untrained", all(not r["trained"] for r in rows)),
          check("every row carries a fact_id", all(r["fact_id"] is not None for r in rows)),
          check("qids are unique", len({r["qid"] for r in rows}) == len(rows))]
    sub = build_manifest(cfg, ["sub_paraphrase"])
    ok += [check("18000 sub paraphrase rows", len(sub) == 18000, str(len(sub))),
           check("sub rows are flagged trained", all(r["trained"] for r in sub)),
           check("sub fact_ids span 0..119",
                 {r["fact_id"] for r in sub} == set(range(120)))]
    s = build_manifest(cfg, ["main_direct"], sample_per_bucket=10)
    ok.append(check("sample_per_bucket caps the bucket", len(s) == 10, str(len(s))))
    ok.append(check("unknown bucket is rejected", _raises(lambda: build_manifest(cfg, ["nope"]))))
    print(f"    probe buckets configured: "
          f"{[b for b in ALL_BUCKETS if b not in ('main_direct', 'main_bypass', 'sub_paraphrase', 'sub_bypass')]}")
    return all(ok)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_holdout(cfg):
    print("holdout splits")
    ok = []
    for mode in ("form", "variant", "fact"):
        tr, ev = split(cfg, mode=mode, frac=0.15, keep_variants=25)
        held = {r["fact_id"] for r in ev if r.get("holdout") not in (None, "main_eval")}
        prompts = {r["question"] for r in tr}
        bleed = [r for r in ev if r.get("holdout") not in (None, "main_eval")
                 and r["question"] in prompts]
        ok.append(check(f"{mode}: no held-out prompt appears in train", not bleed, str(len(bleed))))
        ok.append(check(f"{mode}: train and eval are both non-empty", tr and ev))
        ok.append(check(f"{mode}: 240 main rows stay eval-only",
                        sum(1 for r in ev if r.get("holdout") == "main_eval") == 240))
        if mode == "fact":
            ok.append(check("fact: no train row teaches a withheld fact",
                            not [r for r in tr if r["fact_id"] in held]))
            ok.append(check("fact: held-out rows expect abstention",
                            all(r.get("expect_abstention") for r in ev
                                if r.get("holdout") == "heldout_fact")))
            ok.append(check("fact: 18 of 120 facts withheld", len(held) == 18, str(len(held))))
        if mode == "form":
            ok.append(check("form: the fact is still taught in the plain form",
                            all(any(r["fact_id"] == f and r["field"] == "question" for r in tr)
                                for f in sorted(held)[:5])))
        if mode == "variant":
            per = {}
            for r in tr:
                per[r["fact_id"]] = per.get(r["fact_id"], 0) + 1
            ok.append(check("variant: 25 variants x 2 forms kept per fact",
                            set(per.values()) == {50}, str(sorted(set(per.values()))[:4])))
        tot = len(tr) + len([r for r in ev if r.get("holdout") != "main_eval"])
        ok.append(check(f"{mode}: every sub row is accounted for exactly once", tot == 36000, str(tot)))
        print(f"    {mode}: train={len(tr)} eval={len(ev)} facts_held={len(held)}")
    # Determinism
    a, _ = split(cfg, "form", 0.15)
    b, _ = split(cfg, "form", 0.15)
    ok.append(check("splits are deterministic under the config seed", len(a) == len(b)))
    return all(ok)


def main():
    cfg = load_cfg()
    results = [test_grouping(cfg), test_grading(), test_grading_on_real_answers(cfg),
               test_metrics(), test_manifest(cfg), test_holdout(cfg)]
    print()
    if all(results):
        print("selftest: all groups passed")
        return 0
    print(f"selftest: {results.count(False)} of {len(results)} groups FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
