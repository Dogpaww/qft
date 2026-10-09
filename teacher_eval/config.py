"""Config access for the evaluation package.

Reads the repo's single `config.yaml` -- the project's rule is that every
hyperparameter, path and threshold lives there. Defaults for the new `eval:`
section are kept here as well so the package runs against an unmodified
config.yaml; anything present in the file wins over these.
"""
import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "config.yaml")

DEFAULTS = {
    "manifest": "data/eval/manifest.jsonl",
    "answers": "data/eval/answers.jsonl",
    "graded": "data/eval/graded.jsonl",
    "report_dir": "data/eval",
    "teacher_answers": "data/eval/teacher_answers.jsonl",
    "model_source": "adapter",          # adapter | merged | base
    "batch_size": 16,
    "max_new_tokens": 100,
    "sample_per_bucket": 0,             # 0 = every question in the bucket
    "buckets": ["main_direct", "main_bypass", "general_knowledge",
                "unanswerable", "false_premise"],
    "grading": {
        "closeness_threshold": 0.92,
        "abstention_phrases": None,     # None = use grading.DEFAULT_ABSTENTION_PHRASES
    },
    "probes": {
        "general_knowledge": "data/raw_data/probes_general.json",
        "unanswerable": "data/raw_data/probes_unanswerable.json",
        "false_premise": "data/raw_data/probes_false_premise.json",
    },
    "holdout": {
        "mode": "form",
        "frac": 0.15,
        "keep_variants": 25,
        "out_train": "data/model_data/qa_train_holdout.jsonl",
        "out_eval": "data/model_data/qa_eval_holdout.jsonl",
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_cfg(path=None):
    with open(path or CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["eval"] = _merge(DEFAULTS, cfg.get("eval"))
    return cfg


def resolve(cfg, key):
    """Absolute path for an eval path key, relative to the repo root."""
    p = cfg["eval"][key]
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
