"""Evaluation framework and student-facing teacher pass for the lore teacher.

The finished parts of this project are the seven flat files at the repo root
(`data_prep.py`, `train.py`, `check_no_rag.py`, `evaluate.py`, plus config,
requirements and README). Those are left alone. This package holds only the
work the README still lists as unbuilt:

  buckets.py       bucket taxonomy + the eval manifest, with an exact fact grouping
  grading.py       three-way grading scheme (correct / incorrect / abstained)
  metrics.py       accuracy, hallucination, abstention, teacher-student agreement
  holdout.py       fact-level splits -- the generalisation measurement the pivot dropped
  runner.py        closed-book batched generation (needs a GPU)
  report.py        grade an answers file and render markdown + plots
  teacher_pass.py  resumable teacher pass + gold audit -> the student distillation set
  selftest.py      offline checks for everything that does not need the model

`evaluate.py` at the root stays as the quick smoke test it was built to be.
This package is the full pass that replaces it for real measurement.
"""
__all__ = ["buckets", "config", "grading", "holdout", "metrics", "report",
           "runner", "teacher_pass"]
