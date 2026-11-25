#!/usr/bin/env python3
"""Fixture RAG pipeline for the eval-noise study (stdlib + numpy/torch/sentence-transformers/mlx-lm).

    python3 rag.py selftest   # offline check of query building + rank fusion
    python3 rag.py index      # BM25 (SQLite FTS5) + dense (bge-base, 768-d) indexes -> data/index/
    python3 rag.py recall     # Recall@k sanity check on the frozen eval set
    python3 rag.py generate   # ONE frozen answer per eval question -> data/answers.jsonl

Run from the folder that contains data/ (created by setup_clapnq.py).
Nothing here is tuned: hybrid = reciprocal-rank fusion of BM25 and dense, top-3 passages to the generator.
"""
import csv, io, json, re, sqlite3, sys, time, zipfile
from pathlib import Path

DATA = Path("data")
IDX = DATA / "index"
EMB, EMB_REV = "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
GEN, GEN_REV = "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "a4b8f870474b0eb527f466a03fbc187830d271f5"
QPREFIX = "Represent this sentence for searching relevant passages: "
POOL, RRF_K = 50, 60      # candidates per retriever, RRF constant
K_RETR, K_CTX = 10, 3     # stored for Recall@k, passed to generator
SEED, SHARD = 1234, 20000
ABSTAIN = "I cannot answer this from the provided passages."


# ---------- pure helpers (covered by selftest) ----------
def bm25_query(q):
    """Quote every token so words like AND/OR/NOT/NEAR can't be parsed as FTS5 operators."""
    return " OR ".join('"%s"' % w for w in re.findall(r"\w+", q.lower()))


def fuse(*ranked):
    """Reciprocal-rank fusion over ranked id lists; ties broken by id for determinism."""
    s = {}
    for lst in ranked:
        for r, i in enumerate(lst, 1):
            s[i] = s.get(i, 0.0) + 1.0 / (RRF_K + r)
    return sorted(s, key=lambda i: (-s[i], i))


def selftest():
    db = sqlite3.connect(":memory:")
    db.execute("create virtual table p using fts5(title, text, tokenize='porter unicode61')")
    for i, t in enumerate(["the cat sat on the mat", "dogs and cats are pets", "stock markets fell"]):
        db.execute("insert into p(rowid,title,text) values(?,?,?)", (i, "", t))
    hit = lambda q: [r[0] for r in db.execute(
        "select rowid from p where p match ? order by bm25(p)", (bm25_query(q),))]
    assert hit("what is NOT a cat?")[0] in (0, 1)   # operator word must not raise
    assert hit("cats") and hit("stocks") == [2]     # porter stemming works
    assert bm25_query("???") == ""                  # caller must skip empty queries
    assert fuse(["a", "b"], ["b", "c"])[0] == "b"   # appears in both lists
    assert fuse(["x"], ["y"]) == ["x", "y"]         # deterministic tie-break
    print("selftest ok")


# ---------- data ----------
def load_corpus():
    with zipfile.ZipFile(DATA / "raw" / "passages.tsv.zip") as z:
        name = next(n for n in z.namelist() if n.endswith(".tsv"))
        rows = csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8"), delimiter="\t")
        return [(r["id"], r["title"], r["text"]) for r in rows]


def hf_path(repo, rev, patterns):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, allow_patterns=patterns)


def embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    p = hf_path(EMB, EMB_REV, ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
    m = SentenceTransformer(p, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 256
    return m


# ---------- index ----------
def build_index():
    import numpy as np
    IDX.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    (IDX / "ids.json").write_text(json.dumps([c[0] for c in corpus]))

    db_path = IDX / "bm25.db"
    if not db_path.exists():
        tmp = sqlite3.connect(IDX / "bm25.tmp")
        tmp.execute("create virtual table p using fts5(title, text, tokenize='porter unicode61')")
        tmp.executemany("insert into p(rowid,title,text) values(?,?,?)",
                        ((i, t, x) for i, (_, t, x) in enumerate(corpus)))
        tmp.commit(); tmp.close()
        (IDX / "bm25.tmp").rename(db_path)
        print("bm25 index built")

    if not (IDX / "emb.npy").exists():
        m = embedder()
        shards = []
        for s in range(0, len(corpus), SHARD):   # shard files make an interrupted run resumable
            f = IDX / f"emb_{s:07d}.npy"
            if not f.exists():
                t0 = time.time()
                texts = [f"{t}\n{x}" for _, t, x in corpus[s:s + SHARD]]
                np.save(f, m.encode(texts, batch_size=64, normalize_embeddings=True).astype("float16"))
                print(f"embedded {s + len(texts)}/{len(corpus)} ({time.time() - t0:.0f}s)", flush=True)
            shards.append(f)
        np.save(IDX / "emb.npy", np.concatenate([np.load(f) for f in shards]))
        for f in shards:
            f.unlink()
    print("index ready:", IDX)


class Retriever:
    def __init__(self):
        import numpy as np
        self.np = np
        self.ids = json.loads((IDX / "ids.json").read_text())
        self.text = {i: (t, x) for i, t, x in load_corpus()}
        self.emb = np.load(IDX / "emb.npy").astype("float32")
        self.db = sqlite3.connect(IDX / "bm25.db")
        self.model = embedder()

    def search(self, q, exclude=(), k=K_RETR):
        """Hybrid top-k passage ids. `exclude` removes ids (used for drop_gold items)."""
        ex = set(exclude)
        mq = bm25_query(q)
        sparse = [self.ids[r[0]] for r in self.db.execute(
            "select rowid from p where p match ? order by bm25(p) limit ?", (mq, POOL + len(ex)))] if mq else []
        qv = self.model.encode(QPREFIX + q, normalize_embeddings=True)
        order = self.np.argsort(-(self.emb @ qv), kind="stable")[:POOL + len(ex)]
        dense = [self.ids[i] for i in order]
        keep = lambda lst: [i for i in lst if i not in ex][:POOL]
        return fuse(keep(sparse), keep(dense))[:k]


def eval_items():
    return [json.loads(l) for l in (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]


def recall():
    r = Retriever()
    ks, hits, n = (1, 3, 5, 10), dict.fromkeys((1, 3, 5, 10), 0), 0
    for x in eval_items():
        if not x["answerable"]:
            continue
        got = r.search(x["question"], exclude=x["gold_passage_ids"] if x["drop_gold"] else ())
        if x["drop_gold"]:
            assert not set(got) & set(x["gold_passage_ids"])
            continue
        n += 1
        for k in ks:
            hits[k] += bool(set(got[:k]) & set(x["gold_passage_ids"]))
    print(f"n={n} answerable, gold kept in index")
    print("  ".join(f"Recall@{k}={hits[k] / n:.3f}" for k in ks))


# ---------- generation ----------
def prompt(q, ctx):
    ps = "\n\n".join(f"Passage {i} ({t}): {x}" for i, (_, t, x) in enumerate(ctx, 1))
    return ("Answer the question using only the passages below. Be concise (2-4 sentences). "
            f'If the passages do not contain the answer, reply exactly: "{ABSTAIN}"\n\n'
            f"{ps}\n\nQuestion: {q}")


def generate():
    import mlx.core as mx
    from mlx_lm import generate as gen, load
    from mlx_lm.sample_utils import make_sampler
    out = DATA / "answers.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    r = Retriever()
    model, tok = load(hf_path(GEN, GEN_REV, ["*.json", "*.model", "*.safetensors", "tokenizer*"]))
    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)   # greedy: the fixed layer must not vary
    items = [x for x in eval_items() if x["id"] not in done]
    for n, x in enumerate(items, 1):
        t0 = time.time()
        ids = r.search(x["question"], exclude=x["gold_passage_ids"] if x["drop_gold"] else ())
        ctx = [(i, *r.text[i]) for i in ids[:K_CTX]]
        chat = tok.apply_chat_template([{"role": "user", "content": prompt(x["question"], ctx)}],
                                       add_generation_prompt=True, tokenize=False)
        ans = gen(model, tok, prompt=chat, max_tokens=200, sampler=sampler).strip()
        rec = dict(id=x["id"], question=x["question"], answerable=x["answerable"], drop_gold=x["drop_gold"],
                   retrieved_ids=ids, contexts=[dict(id=i, title=t, text=s) for i, t, s in ctx],
                   answer=ans, gen_model=GEN, gen_rev=GEN_REV, emb_model=EMB, emb_rev=EMB_REV,
                   seed=SEED, temp=0.0, seconds=round(time.time() - t0, 2))
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{n}/{len(items)} {rec['seconds']}s | {ans[:80]!r}", flush=True)


if __name__ == "__main__":
    cmds = dict(selftest=selftest, index=build_index, recall=recall, generate=generate)
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
