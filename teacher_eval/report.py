"""Grade an answers file and render the results.

Separated from `runner.py` on purpose: grading is cheap and deterministic, so a
finished answers file can be re-graded with a changed threshold or a new
abstention phrase without spending GPU time again.

    python -m teacher_eval.report                    # grade + render
    python -m teacher_eval.report --answers path.jsonl --compare base.jsonl
"""
import argparse
import json
import os

from . import metrics
from .config import ROOT, load_cfg, resolve
from .grading import DEFAULT_ABSTENTION_PHRASES, grade


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def grade_rows(cfg, rows):
    gc = cfg["eval"]["grading"]
    phrases = tuple(gc.get("abstention_phrases") or DEFAULT_ABSTENTION_PHRASES)
    out = []
    for r in rows:
        v = grade(r.get("gold", ""), r.get("answer", ""),
                  expect_abstention=bool(r.get("expect_abstention")),
                  closeness_threshold=gc["closeness_threshold"],
                  abstention_phrases=phrases, aliases=r.get("aliases") or ())
        out.append(dict(r, **v))
    return out


def _pct(x):
    return f"{x * 100:.1f}%"


def render_markdown(cfg, graded, compare=None):
    ov = metrics.overall(graded)
    bb = metrics.by_bucket(graded)
    src = graded[0].get("model_source", "?") if graded else "?"
    L = [f"# Teacher evaluation ({src})", "",
         f"Base model: `{cfg['models']['base']}`  |  graded: {ov['n']} questions", "",
         "Verdicts are three-way: `correct`, `incorrect` (a confident wrong answer -- "
         "the hallucination signal) and `abstained` (declined). The three sum to 1 "
         "per bucket. Intervals are 95% Wilson.", "",
         "## By bucket", "",
         "| bucket | n | accuracy | 95% CI | hallucination | abstention | trained on? |",
         "|---|---|---|---|---|---|---|"]
    for b, m in bb.items():
        lo, hi = m["accuracy_ci95"]
        trained = {True: "yes -- memorisation", False: "no", "mixed": "mixed"}[m["trained"]]
        L.append(f"| `{b}` | {m['n']} | {_pct(m['accuracy'])} | "
                 f"{_pct(lo)}-{_pct(hi)} | {_pct(m['hallucination'])} | "
                 f"{_pct(m['abstention'])} | {trained} |")
    L += ["", f"**Overall accuracy** {_pct(ov['accuracy'])} "
              f"({ov['correct']}/{ov['n']}), hallucination {_pct(ov['hallucination'])}, "
              f"abstention {_pct(ov['abstention'])}.", ""]

    rules = metrics.by_rule(graded)
    if rules:
        L += ["## Which rule scored each correct answer", "",
              "A score carried by `token_contained` is weaker evidence than one carried "
              "by `exact`.", "",
              "| rule | correct answers |", "|---|---|"]
        L += [f"| `{k}` | {v} |" for k, v in rules.items()]
        L.append("")

    weak = metrics.weak_facts(graded)
    L += ["## Facts not fully learned", ""]
    if not weak:
        L.append("Every fact with a `fact_id` scored 100% in every bucket it appears in.")
    else:
        L += [f"{len(weak)} of the facts scored below 100%.", "",
              "| fact_id | gold | n | accuracy | example missed prompt |", "|---|---|---|---|---|"]
        for w in weak[:40]:
            ex = w["example_miss"].replace("|", "\\|")[:80]
            L.append(f"| {w['fact_id']} | {w['gold'][:40]} | {w['n']} | "
                     f"{_pct(w['accuracy'])} | {ex} |")
    L.append("")

    if compare:
        agr = metrics.agreement(compare, graded, lambda g, a: grade(g, a))
        L += ["## Teacher-to-student answer checking", "",
              f"Paired on `qid`, {agr['n']} questions.", "",
              f"- matched the teacher's answer: {_pct(agr['matched_teacher'])}",
              f"- both correct against gold: {_pct(agr['both_correct'])}",
              f"- student accuracy: {_pct(agr['student_accuracy'])}",
              f"- teacher accuracy: {_pct(agr['teacher_accuracy'])}", "",
              "Matching the teacher and being correct are different things; the "
              "competition scores the first.", ""]
    return "\n".join(L)


def plot_buckets(graded, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bb = metrics.by_bucket(graded)
    names = list(bb)
    acc = [bb[b]["accuracy"] for b in names]
    hal = [bb[b]["hallucination"] for b in names]
    abst = [bb[b]["abstention"] for b in names]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4))
    bottom = [0.0] * len(names)
    for vals, label in ((acc, "correct"), (hal, "incorrect"), (abst, "abstained")):
        ax.bar(names, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 1); ax.set_ylabel("share of bucket")
    ax.legend(loc="lower right"); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(path)
    return path


def run(cfg, answers_path=None, compare_path=None):
    answers_path = answers_path or resolve(cfg, "answers")
    if not os.path.exists(answers_path):
        raise SystemExit(f"no answers at {answers_path}; run `python -m teacher_eval.runner` first")
    graded = grade_rows(cfg, read_jsonl(answers_path))

    graded_path = resolve(cfg, "graded")
    out_dir = resolve(cfg, "report_dir")
    os.makedirs(out_dir, exist_ok=True)
    with open(graded_path, "w", encoding="utf-8") as f:
        for r in graded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    compare = grade_rows(cfg, read_jsonl(compare_path)) if compare_path else None
    md = render_markdown(cfg, graded, compare)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    json.dump({"overall": metrics.overall(graded), "by_bucket": metrics.by_bucket(graded),
               "by_rule": metrics.by_rule(graded)},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    try:
        plot_buckets(graded, os.path.join(out_dir, "buckets.png"))
    except Exception as e:                           # a missing backend must not lose the report
        print("plot skipped:", e)
    print(md)
    print(f"\nwritten: {md_path}, {graded_path}, {out_dir}/summary.json")
    return md_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--answers", default=None)
    ap.add_argument("--compare", default=None,
                    help="a second answers file to treat as the teacher for agreement")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    run(load_cfg(a.config), a.answers, a.compare)


if __name__ == "__main__":
    main()
