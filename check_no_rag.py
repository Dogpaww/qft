"""No-lookup guard. Fails loudly (exit 1 / AssertionError) if lookup machinery or lore-in-prompt appears.

  python check_no_rag.py          # scan repo + verify the inference prompt
Any external eval/inference harness should call run_guard() first and assert_student_messages() before every generation.
"""
import ast, glob, json, os, re, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FORBIDDEN_MODULES = {
    "faiss", "chromadb", "qdrant_client", "pinecone", "weaviate", "pymilvus", "lancedb", "annoy", "hnswlib",
    "elasticsearch", "whoosh", "rank_bm25", "bm25s", "sentence_transformers", "langchain", "langchain_core",
    "langchain_community", "llama_index", "haystack", "txtai", "sklearn",
}
FORBIDDEN_WORDS = re.compile(
    r"retriev|vector[_ ]?(store|db|database)|vectorstore|bm25|tf-?idf|cosine_similarity|embedding_search|"
    r"top_k_docs|similarity_search|knn_search|inject_lore|context_docs", re.I)


def fail(msg):
    print(f"\n*** NO-RAG GUARD FAILED: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def scan_source():
    me = os.path.abspath(__file__)
    for p in glob.glob(os.path.join(HERE, "*.py")) + [os.path.join(HERE, "requirements.txt")]:
        if os.path.abspath(p) == me or not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        if p.endswith(".py"):
            for node in ast.walk(ast.parse(src)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for n in names:
                    if n.split(".")[0] in FORBIDDEN_MODULES:
                        fail(f"{os.path.basename(p)} imports forbidden module '{n}'")
        else:
            for line in src.lower().splitlines():
                pkg = re.split(r"[=<>\[ #]", line.strip())[0].replace("-", "_")
                if pkg in FORBIDDEN_MODULES:
                    fail(f"requirements.txt lists forbidden package '{pkg}'")
        m = FORBIDDEN_WORDS.search(src)
        if m:
            fail(f"{os.path.basename(p)} contains forbidden keyword '{m.group(0)}'")


def _ngrams(text, n):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_prompt(cfg):
    sp, nc = cfg["inference"]["system_prompt"], cfg["no_rag"]
    if len(sp) > nc["max_system_prompt_chars"]:
        fail(f"system prompt is {len(sp)} chars (> {nc['max_system_prompt_chars']}); lore may be hiding in it")
    paths = cfg["paths"]
    if os.path.exists(paths["graph"]):
        graph = json.load(open(paths["graph"], encoding="utf-8"))
        low = sp.lower()
        for e in graph["entities"]:
            if len(e) > 3 and re.search(rf"\b{re.escape(e.lower())}\b", low):
                fail(f"system prompt mentions lore entity '{e}'")
        for f in graph["facts"].values():
            if _ngrams(sp, nc["ngram_overlap"]) & _ngrams(f["text"], nc["ngram_overlap"]):
                fail("system prompt shares a long n-gram with a lore fact")
    if os.path.exists(paths["corpus"]):
        if _ngrams(sp, nc["ngram_overlap"]) & _ngrams(open(paths["corpus"], encoding="utf-8").read(), nc["ngram_overlap"]):
            fail("system prompt shares a long n-gram with the corpus")


def assert_student_messages(messages, cfg):
    """Student input must be exactly [fixed lore-free system prompt, bare question]. Nothing else."""
    ok = (len(messages) == 2 and messages[0] == {"role": "system", "content": cfg["inference"]["system_prompt"]}
          and messages[1]["role"] == "user" and len(messages[1]["content"]) <= cfg["no_rag"]["max_question_chars"])
    if not ok:
        fail(f"student prompt is not [fixed system prompt, bare question]: {str(messages)[:200]}")


def run_guard(cfg):
    scan_source()
    check_prompt(cfg)
    print("no-RAG guard: OK")


if __name__ == "__main__":
    run_guard(yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8")))
