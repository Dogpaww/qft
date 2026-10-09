"""Aggregate graded rows into the metrics the README asks for.

README "Next features" names: golden-truth accuracy, hallucination rate,
teacher-to-student answer checking, reported per bucket. This module computes
those and nothing else -- it takes already-graded rows and returns numbers, so
it can be tested without a model.

Definitions used here, stated explicitly because the names are ambiguous:

  accuracy          correct / graded                 (the golden-truth score)
  hallucination     incorrect / graded               a confident answer that is
                                                     wrong. For the unanswerable
                                                     and false_premise buckets a
                                                     confident answer is itself
                                                     the failure, so answering
                                                     counts here too.
  abstention        abstained / graded               declined to answer
  coverage          1 - abstention                   share of questions answered

accuracy + hallucination + abstention == 1 for every bucket.

Small samples get a 95% Wilson interval: the main buckets are 120 questions, so
a bare percentage invites over-reading. Wilson is used rather than the normal
approximation because it stays sane at 0/120 and 120/120 -- both of which this
model card already reports.
"""
import math

from .grading import VERDICT_ABSTAINED, VERDICT_CORRECT, VERDICT_INCORRECT


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _tally(rows):
    n = len(rows)
    c = sum(1 for r in rows if r["verdict"] == VERDICT_CORRECT)
    i = sum(1 for r in rows if r["verdict"] == VERDICT_INCORRECT)
    a = sum(1 for r in rows if r["verdict"] == VERDICT_ABSTAINED)
    lo, hi = wilson(c, n)
    return {"n": n, "correct": c, "incorrect": i, "abstained": a,
            "accuracy": c / n if n else 0.0,
            "hallucination": i / n if n else 0.0,
            "abstention": a / n if n else 0.0,
            "coverage": 1 - (a / n) if n else 0.0,
            "accuracy_ci95": [round(lo, 4), round(hi, 4)]}


def by_bucket(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["bucket"], []).append(r)
    out = {b: _tally(rs) for b, rs in sorted(groups.items())}
    for b, rs in groups.items():
        seen = sum(1 for r in rs if r.get("trained"))
        # Flagged so a report can never present a memorisation score as a
        # generalisation score.
        out[b]["trained"] = True if seen == len(rs) else (False if seen == 0 else "mixed")
    return out


def overall(rows):
    return _tally(rows)


def by_rule(rows):
    """How many correct answers each grading rule produced.

    Worth reading: a score carried by `token_contained` rather than `exact` is a
    weaker result than the headline number suggests.
    """
    out = {}
    for r in rows:
        if r["verdict"] == VERDICT_CORRECT:
            out[r["rule"]] = out.get(r["rule"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def weak_facts(rows, threshold=1.0):
    """Facts whose accuracy is below `threshold`, worst first.

    This is the actionable output: it names which of the 120 facts the model
    did not learn, instead of reporting one aggregate number.
    """
    groups = {}
    for r in rows:
        if r.get("fact_id") is not None:
            groups.setdefault(r["fact_id"], []).append(r)
    out = []
    for fid, rs in groups.items():
        acc = sum(1 for r in rs if r["verdict"] == VERDICT_CORRECT) / len(rs)
        if acc < threshold:
            out.append({"fact_id": fid, "n": len(rs), "accuracy": acc,
                        "gold": rs[0].get("gold", ""),
                        "example_miss": next((r.get("prompt", "") for r in rs
                                              if r["verdict"] != VERDICT_CORRECT), "")})
    return sorted(out, key=lambda d: (d["accuracy"], d["fact_id"]))


def agreement(teacher_rows, student_rows, grade_fn):
    """Teacher-to-student answer checking.

    Grades the student's answer against the *teacher's* answer as gold, matched
    on qid, and also reports how often both were right against the true gold --
    the competition cares about matching the teacher, which is not the same as
    being correct.
    """
    t = {r["qid"]: r for r in teacher_rows}
    pairs = [(t[r["qid"]], r) for r in student_rows if r["qid"] in t]
    if not pairs:
        return {"n": 0, "matched_teacher": 0.0, "both_correct": 0.0,
                "student_accuracy": 0.0, "teacher_accuracy": 0.0}
    match = sum(1 for tr, sr in pairs
                if grade_fn(tr.get("answer", ""), sr.get("answer", ""))["verdict"] == VERDICT_CORRECT)
    both = sum(1 for tr, sr in pairs
               if tr["verdict"] == VERDICT_CORRECT and sr["verdict"] == VERDICT_CORRECT)
    return {"n": len(pairs),
            "matched_teacher": match / len(pairs),
            "both_correct": both / len(pairs),
            "student_accuracy": sum(1 for _, sr in pairs if sr["verdict"] == VERDICT_CORRECT) / len(pairs),
            "teacher_accuracy": sum(1 for tr, _ in pairs if tr["verdict"] == VERDICT_CORRECT) / len(pairs)}
