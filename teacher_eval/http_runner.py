"""Closed-book generation against a hosted teacher endpoint.

Same job as `runner.py`, but over HTTP instead of local weights, so the full
evaluation can run without a GPU. Standard library only (urllib), to keep the
dependency set as lean as the rest of the pipeline.

The no-lookup policy is enforced exactly as elsewhere: the repo-wide scan runs
once at start-up, and `assert_student_messages` runs on every prompt before it
leaves the machine, so the served model only ever sees
`[fixed lore-free system prompt, bare question]`.

The API key is never read from source or config. Set LORE_TEACHER_API_KEY, or
pass --key-file. It is sent in an Authorization header, never in the URL, and
is not written to the answers file.

    export LORE_TEACHER_BASE=https://<host>
    python -m teacher_eval.http_runner --key-file ~/Documents/eval.txt
    python -m teacher_eval.http_runner --buckets main_direct --workers 8
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .buckets import build_manifest, counts
from .config import ROOT, load_cfg, resolve

DEFAULT_TIMEOUT = 90


def _post(url, payload, key, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ask(base, key, system, question, model="lore-teacher", max_tokens=100,
        timeout=DEFAULT_TIMEOUT, retries=3):
    """One closed-book question. Returns (answer, latency_seconds, error_or_None)."""
    from check_no_rag import assert_student_messages
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    assert_student_messages(msgs, ask.cfg)          # every prompt, no exceptions
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": msgs}
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            out = _post(base.rstrip("/") + "/v1/chat/completions", payload, key, timeout)
            return out["choices"][0]["message"]["content"].strip(), time.time() - t0, None
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                IndexError, ValueError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return "", 0.0, last


def run(cfg, base, key, buckets=None, sample_per_bucket=None, workers=8, out_path=None):
    from check_no_rag import run_guard
    run_guard(cfg)
    ask.cfg = cfg

    ec = cfg["eval"]
    rows = build_manifest(cfg, buckets or ec["buckets"],
                          ec["sample_per_bucket"] if sample_per_bucket is None else sample_per_bucket)
    if not rows:
        raise SystemExit("manifest is empty; check eval.buckets and the probe files")
    print("manifest:", counts(rows), f"total={len(rows)}")
    system = cfg["inference"]["system_prompt"]

    done = [0]

    def one(r):
        a, lat, err = ask(base, key, system, r["prompt"], max_tokens=ec["max_new_tokens"])
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(rows):
            print(f"  {done[0]}/{len(rows)}", flush=True)
        return dict(r, answer=a, latency_s=round(lat, 3), error=err,
                    model_source=f"endpoint:{base.rstrip('/').split('//')[-1].split('.')[0]}")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, rows))
    elapsed = time.time() - t0

    path = out_path or resolve(cfg, "answers")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errs = [r for r in out if r["error"]]
    lats = sorted(r["latency_s"] for r in out if not r["error"])
    print(f"\n{len(out)} answers in {elapsed:.1f}s -> {path}")
    if lats:
        print(f"latency: median {lats[len(lats) // 2]:.2f}s  "
              f"p95 {lats[int(len(lats) * 0.95)]:.2f}s  max {lats[-1]:.2f}s")
    if errs:
        print(f"WARNING: {len(errs)} requests failed, e.g. {errs[0]['error'][:120]}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=os.environ.get("LORE_TEACHER_BASE"),
                    help="endpoint root, or set LORE_TEACHER_BASE")
    ap.add_argument("--key-file", default=None, help="file holding the API key")
    ap.add_argument("--buckets", nargs="*", default=None)
    ap.add_argument("--sample-per-bucket", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if not a.base:
        raise SystemExit("no endpoint: pass --base or set LORE_TEACHER_BASE")
    key = os.environ.get("LORE_TEACHER_API_KEY", "")
    if a.key_file:
        with open(os.path.expanduser(a.key_file), encoding="utf-8") as f:
            key = f.read().strip()
    os.chdir(ROOT)
    run(load_cfg(a.config), a.base, key, a.buckets, a.sample_per_bucket, a.workers, a.out)


if __name__ == "__main__":
    main()
