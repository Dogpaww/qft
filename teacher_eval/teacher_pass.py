"""Run the fine-tuned teacher over a question set to produce answers for students.

The README lists this as a pending stage ("run the fine-tuned teacher over the
questions to produce answers for students"). It is the competition deliverable:
contestants train smaller student models to match these answers, so this file
is what gets distributed, not the eval report.

Two things it does beyond plain generation:

Resumable.  The full set is 36,000 prompts (18,000 variants x 2 forms) and runs
for hours. Answers are appended as they are produced and already-answered
`qid`s are skipped on a re-run, so an interrupted pass costs only the work it
had not reached.

Audited against gold.  Every teacher answer is graded against the known gold
answer before distribution and the disagreements are written to a separate
file. A teacher answer that contradicts gold would otherwise be taught to every
student as ground truth. Nothing is filtered automatically -- the mismatches
are reported for a decision, since the gold string can also be the narrow one.

    python -m teacher_eval.teacher_pass --buckets sub_paraphrase sub_bypass
    python -m teacher_eval.teacher_pass --resume
"""
import argparse
import json
import os

from .buckets import build_manifest, counts
from .config import ROOT, load_cfg, resolve
from .grading import VERDICT_CORRECT, grade

DISTILL_FIELDS = ("qid", "bucket", "fact_id", "form", "question", "teacher_answer")


def _done_qids(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.add(json.loads(line)["qid"])
                except (ValueError, KeyError):
                    continue          # a half-written final line from a kill
    return out


def run(cfg, buckets=None, resume=False, batch_size=None, out_path=None):
    from check_no_rag import run_guard
    from .runner import _load_model, generate

    run_guard(cfg)
    ec = cfg["eval"]
    out_path = out_path or resolve(cfg, "teacher_answers")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = build_manifest(cfg, buckets or ["sub_paraphrase", "sub_bypass"], 0)
    done = _done_qids(out_path) if resume else set()
    todo = [r for r in rows if r["qid"] not in done]
    print(f"manifest: {counts(rows)}")
    print(f"{len(done)} already answered, {len(todo)} to go")
    if not todo:
        print("nothing to do")
        return out_path

    tok, model = _load_model(cfg, ec["model_source"])
    bs = batch_size or ec["batch_size"]
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            answers = generate(cfg, tok, model, [r["prompt"] for r in chunk], batch_size=bs)
            for r, a in zip(chunk, answers):
                f.write(json.dumps({"qid": r["qid"], "bucket": r["bucket"],
                                    "fact_id": r["fact_id"], "form": r["form"],
                                    "question": r["prompt"], "teacher_answer": a,
                                    "gold": r["gold"]}, ensure_ascii=False) + "\n")
            f.flush()                 # survive a kill mid-pass
    print(f"teacher answers -> {out_path}")
    return out_path


def audit(cfg, path=None):
    """Grade the teacher's own answers against gold; write the disagreements."""
    path = path or resolve(cfg, "teacher_answers")
    if not os.path.exists(path):
        raise SystemExit(f"no teacher answers at {path}; run the pass first")
    gc = cfg["eval"]["grading"]
    rows, bad = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = grade(r.get("gold", ""), r.get("teacher_answer", ""),
                      closeness_threshold=gc["closeness_threshold"])
            rows.append(v)
            if v["verdict"] != VERDICT_CORRECT:
                bad.append(dict(r, **v))
    n = len(rows)
    ok = sum(1 for v in rows if v["verdict"] == VERDICT_CORRECT)
    out = os.path.join(os.path.dirname(path), "teacher_mismatches.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"teacher agrees with gold on {ok}/{n}" + (f" = {ok / n:.2%}" if n else ""))
    print(f"{len(bad)} mismatches -> {out}")
    if bad:
        print("review these before distributing; a teacher answer that contradicts gold "
              "would be taught to every student as ground truth")
    return {"n": n, "agree": ok, "mismatches": len(bad)}


def write_distill_set(cfg, path=None, out=None, drop_mismatches=False):
    """Strip gold and write the file students actually receive."""
    path = path or resolve(cfg, "teacher_answers")
    out = out or os.path.join(os.path.dirname(path), "distill_set.jsonl")
    gc = cfg["eval"]["grading"]
    kept = dropped = 0
    with open(path, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            r = json.loads(line)
            if drop_mismatches and grade(r.get("gold", ""), r.get("teacher_answer", ""),
                                         closeness_threshold=gc["closeness_threshold"]
                                         )["verdict"] != VERDICT_CORRECT:
                dropped += 1
                continue
            dst.write(json.dumps({k: r.get(k) for k in DISTILL_FIELDS},
                                 ensure_ascii=False) + "\n")
            kept += 1
    print(f"{kept} rows -> {out}" + (f" ({dropped} mismatches dropped)" if dropped else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--buckets", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--distill-set", action="store_true",
                    help="write the student-facing file from an existing pass")
    ap.add_argument("--drop-mismatches", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_cfg(a.config)
    if a.audit_only:
        audit(cfg, a.out)
    elif a.distill_set:
        write_distill_set(cfg, a.out, drop_mismatches=a.drop_mismatches)
    else:
        run(cfg, a.buckets, a.resume, a.batch_size, a.out)
        audit(cfg, a.out)


if __name__ == "__main__":
    main()
