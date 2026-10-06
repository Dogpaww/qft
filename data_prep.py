"""Stages 1-3: clean corpus -> knowledge graph -> teacher QA -> fact-level split.

Usage (each stage is resumable and writes artifacts to paths.out_dir):
  python data_prep.py clean            # corpus -> chunks.jsonl
  python data_prep.py graph            # chunks -> graph.json (teacher extracts facts)
  python data_prep.py qa [--dry-run]   # graph -> qa_all.jsonl (teacher; costs money)
  python data_prep.py split            # qa_all -> qa_train / qa_eval (by fact_id)
  python data_prep.py sample           # print a few graph facts and QA pairs
"""
import argparse, hashlib, json, os, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import yaml

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_cfg(path=CFG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- teacher helper ----------------
def teacher_json(cfg, system, user, retries=3):
    """PLACEHOLDER: teacher model call. Not implemented yet (no provider/API key chosen).

    Contract: send `system` + `user` to the teacher model named in cfg["models"]["teacher"],
    parse the reply as JSON, and return it (dict/list), or None on failure.
    The teacher may see the graph; the student never does.
    """
    raise NotImplementedError("Teacher model not set up: implement teacher_json() in data_prep.py "
                              "and set models.teacher in config.yaml.")


def pmap(cfg, fn, items):
    with ThreadPoolExecutor(cfg["models"]["teacher_workers"]) as ex:
        return list(ex.map(fn, items))


# ---------------- stage 1: clean + chunk ----------------
def stage_clean(cfg):
    raw = open(cfg["paths"]["corpus"], encoding="utf-8").read()
    raw = raw.replace("\r\n", "\n").replace(" ", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    # split on markdown-style headings first (entity/event sections), then pack paragraphs
    sections = re.split(r"\n(?=#{1,4} )", raw)
    chunks, target, lo = [], cfg["ingest"]["chunk_words"], cfg["ingest"]["min_chunk_words"]
    for sec in sections:
        title = sec.split("\n", 1)[0].lstrip("# ").strip()[:80]
        buf = []
        for para in sec.split("\n\n"):
            buf.append(para)
            if sum(len(p.split()) for p in buf) >= target:
                chunks.append({"title": title, "text": "\n\n".join(buf)})
                buf = []
        if buf and sum(len(p.split()) for p in buf) >= lo:
            chunks.append({"title": title, "text": "\n\n".join(buf)})
        elif buf and chunks:
            chunks[-1]["text"] += "\n\n" + "\n\n".join(buf)
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"c{i:04d}"
    write_jsonl(cfg["paths"]["chunks"], chunks)
    print(f"{len(chunks)} chunks, {sum(len(c['text'].split()) for c in chunks)} words")


# ---------------- stage 1b: knowledge graph ----------------
GRAPH_SYS = (
    "You extract a knowledge graph from fictional lore. Return ONLY JSON: "
    '{"facts":[{"text":"one atomic, self-contained statement naming entities explicitly",'
    '"entities":["..."],"relations":[{"s":"..","r":"..","o":".."}],"time":"era/date/order cue or null"}]}. '
    "Use canonical entity names consistently. Every statement must be explicitly supported by the passage."
)


def fact_id(text):
    return "f_" + hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()[:10]


def stage_graph(cfg):
    chunks = read_jsonl(cfg["paths"]["chunks"])

    def one(c):
        out = teacher_json(cfg, GRAPH_SYS, f"Passage titled '{c['title']}':\n\n{c['text']}")
        return [] if not out else [dict(f, chunk_id=c["chunk_id"]) for f in out.get("facts", [])]

    facts, seen = [], set()
    for fl in pmap(cfg, one, chunks):
        for f in fl:
            f["fact_id"] = fact_id(f["text"])
            if f["fact_id"] not in seen:
                seen.add(f["fact_id"])
                facts.append(f)
    ents = defaultdict(list)
    for f in facts:
        for e in f.get("entities", []):
            ents[e].append(f["fact_id"])
    graph = {"facts": {f["fact_id"]: f for f in facts}, "entities": ents}
    json.dump(graph, open(cfg["paths"]["graph"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(facts)} facts, {len(ents)} entities")


# ---------------- stage 2: teacher QA ----------------
BUCKET_SPEC = {
    "direct": "Ask directly for one fact. Answer = that fact. fact_ids = [that fact].",
    "paraphrase": "Ask for one fact using heavily reworded, indirect phrasing (no copied wording from the fact). fact_ids = [that fact].",
    "multihop": "Questions requiring chaining 2+ facts/edges (e.g. the X of the Y of Z). fact_ids = all facts used.",
    "relational": "Questions about how two entities relate / who is linked to whom and how. fact_ids = facts used.",
    "temporal_causal": "Questions on ordering ('before/after'), timing, or cause and effect. fact_ids = facts used.",
    "false_premise": "Questions containing a plausible but FALSE premise or counterfactual about the lore. The answer must reject/correct the premise using the true fact. fact_ids = the contradicted true fact(s).",
    "unanswerable": "Plausible questions about this entity that the lore does NOT answer (details absent from every fact). The answer is exactly the abstention string. fact_ids = [].",
}
QA_SYS = (
    "You write closed-book exam questions about a fictional world from a list of ground-truth facts. "
    "Questions must be self-contained (name entities explicitly; never say 'according to the text' or 'the passage'). "
    "Answers are concise, 1-3 sentences, and fully supported by the facts. "
    'Return ONLY JSON: [{"question":"..","answer":"..","fact_ids":["f_.."],"difficulty":"easy|medium|hard"}]'
)


def plan_requests(cfg, graph):
    qc, rng = cfg["qa"], random.Random(cfg["seed"])
    ents = sorted(graph["entities"], key=lambda e: -len(graph["entities"][e]))[: qc["topics"]]
    fid2ents = {fid: f.get("entities", []) for fid, f in graph["facts"].items()}
    reqs = []
    for e in ents:
        own = graph["entities"][e]
        # neighbours: facts of entities that co-occur with e (for multi-hop / relational)
        nb = {x for fid in own for oe in fid2ents[fid] for x in graph["entities"][oe]} - set(own)
        for bucket, share in qc["bucket_mix"].items():
            n = round(qc["questions_per_topic"] * share)
            pool = own + rng.sample(sorted(nb), min(len(nb), 15)) if bucket in ("multihop", "relational", "temporal_causal") else own
            while n > 0:
                k = min(n, qc["batch_size"])
                facts = rng.sample(pool, min(len(pool), qc["max_facts_in_prompt"]))
                reqs.append({"entity": e, "bucket": bucket, "n": k, "facts": facts})
                n -= k
    return reqs


def stage_qa(cfg, dry_run):
    graph = json.load(open(cfg["paths"]["graph"], encoding="utf-8"))
    reqs = plan_requests(cfg, graph)
    total = sum(r["n"] for r in reqs)
    print(f"{len(reqs)} teacher calls -> ~{total} questions over {cfg['qa']['topics']} topics")
    if dry_run:
        return
    abstain = cfg["qa"]["abstain_answer"]

    def one(r):
        fact_block = "\n".join(f"[{fid}] {graph['facts'][fid]['text']}" for fid in r["facts"])
        user = (f"Entity focus: {r['entity']}\nType: {r['bucket']} - {BUCKET_SPEC[r['bucket']]}\n"
                f"Abstention string (for unanswerable): {abstain}\nWrite {r['n']} questions.\n\nFACTS:\n{fact_block}")
        out = teacher_json(cfg, QA_SYS, user) or []
        rows = []
        for q in out:
            ids = [i for i in q.get("fact_ids", []) if i in graph["facts"]]
            if r["bucket"] != "unanswerable" and not ids:
                continue  # ungrounded -> drop
            rows.append({"question": q["question"].strip(),
                         "answer": abstain if r["bucket"] == "unanswerable" else q["answer"].strip(),
                         "fact_ids": ids, "entity": r["entity"], "bucket": r["bucket"],
                         "difficulty": q.get("difficulty", "medium")})
        return rows

    rows, seen = [], set()
    for rl in pmap(cfg, one, reqs):
        for q in rl:
            if q["question"].lower() not in seen:
                seen.add(q["question"].lower())
                rows.append(q)
    write_jsonl(cfg["paths"]["qa_all"], rows)
    print(f"{len(rows)} QA pairs written")


# ---------------- stage 3: split by fact_id ----------------
def stage_split(cfg):
    sc, rng = cfg["split"], random.Random(cfg["seed"])
    graph = json.load(open(cfg["paths"]["graph"], encoding="utf-8"))
    rows = read_jsonl(cfg["paths"]["qa_all"])
    fids = sorted(graph["facts"])
    rng.shuffle(fids)
    n_held = int(len(fids) * sc["heldout_fact_frac"])
    held = set(fids[:n_held])
    seen_probe = set(fids[n_held: n_held + int(len(fids) * sc["seen_paraphrase_frac"])])
    train, ev = [], []
    for r in rows:
        ids = set(r["fact_ids"])
        if ids & held:
            ev.append(dict(r, split_kind="heldout_fact"))      # generalisation from corpus text only
        elif r["bucket"] == "paraphrase" and ids & seen_probe:
            ev.append(dict(r, split_kind="seen_fact_paraphrase"))  # memorisation probe
        elif r["bucket"] == "unanswerable" and rng.random() < sc["heldout_fact_frac"]:
            ev.append(dict(r, split_kind="heldout_fact"))
        else:
            train.append(r)
    # hard guarantee: no held-out fact appears in any training question
    assert not any(set(r["fact_ids"]) & held for r in train), "leak: held-out fact in train"
    write_jsonl(cfg["paths"]["qa_train"], train)
    write_jsonl(cfg["paths"]["qa_eval"], ev)
    json.dump(sorted(held), open(os.path.join(cfg["paths"]["out_dir"], "heldout_facts.json"), "w"))
    print(f"train={len(train)} eval={len(ev)} heldout_facts={len(held)}")


def stage_sample(cfg):
    g = json.load(open(cfg["paths"]["graph"], encoding="utf-8"))
    for f in random.Random(0).sample(list(g["facts"].values()), min(5, len(g["facts"]))):
        print("FACT", f["fact_id"], f["text"])
    if os.path.exists(cfg["paths"]["qa_all"]):
        for q in random.Random(0).sample(read_jsonl(cfg["paths"]["qa_all"]), 8):
            print(f"QA [{q['bucket']}] {q['question']} -> {q['answer']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["clean", "graph", "qa", "split", "sample"])
    ap.add_argument("--dry-run", action="store_true", help="qa: only count teacher calls")
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    random.seed(cfg["seed"])
    {"clean": stage_clean, "graph": stage_graph, "split": stage_split, "sample": stage_sample,
     "qa": lambda c: stage_qa(c, a.dry_run)}[a.stage](cfg)
