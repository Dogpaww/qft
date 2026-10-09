"""Single entry point for the evaluation framework.

    python -m teacher_eval selftest                 # offline checks, no GPU
    python -m teacher_eval manifest                 # build + summarise the manifest
    python -m teacher_eval holdout --mode form      # generalisation split
    python -m teacher_eval run --source adapter     # generate answers (GPU)
    python -m teacher_eval report                   # grade + render
    python -m teacher_eval teacher-pass --resume    # student distillation set
"""
import os
import sys

from .config import ROOT, load_cfg

COMMANDS = ("selftest", "manifest", "holdout", "run", "report", "teacher-pass")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    os.chdir(ROOT)
    sys.argv = [f"teacher_eval.{cmd}"] + rest

    if cmd == "selftest":
        from .selftest import main as run_cmd
    elif cmd == "manifest":
        from .buckets import build_manifest, counts
        cfg = load_cfg()
        rows = build_manifest(cfg, cfg["eval"]["buckets"], cfg["eval"]["sample_per_bucket"])
        print("manifest:", counts(rows), f"total={len(rows)}")
        for r in rows[:3]:
            print(f"  [{r['bucket']}] fact {r['fact_id']} trained={r['trained']}")
            print(f"    {r['prompt'][:100]}\n    gold: {r['gold']}")
        return 0
    elif cmd == "holdout":
        from .holdout import main as run_cmd
    elif cmd == "run":
        from .runner import main as run_cmd
    elif cmd == "report":
        from .report import main as run_cmd
    else:
        from .teacher_pass import main as run_cmd
    return run_cmd() or 0


if __name__ == "__main__":
    raise SystemExit(main())
