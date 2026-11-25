#!/usr/bin/env python3
"""Judge harness: frozen answers -> constrained Yes/No judgments -> data/judgments/<judge>[_tag].jsonl

    python3 judge.py selftest
    python3 judge.py run --judge qwen2.5-7b [--variants 0,1,2,3] [--n N] [--det-n 100] [--tag pilot]
    python3 judge.py summary data/judgments/<file>.jsonl

Design (differs from RAGAS on purpose; report this in the paper):
  * Every metric is decomposed into single-token Yes/No verdicts. The judge never writes free text, so
    there are no parse failures; `mass` (prob. mass the judge puts on Yes/No) is the format-compliance proxy.
  * Claim/sentence splitting is deterministic (regex), not LLM-generated, so decomposition adds no noise.
  * One forward pass yields d = logit(Yes) - logit(No). The 10 "seed" passes at temperature T are sampled
    from sigmoid(d/T) with a seeded RNG, i.e. they are NOT independent forward passes. Under constrained
    single-token decoding the forward pass is deterministic, so this is equivalent. Real forward-pass
    non-determinism is measured separately (T=0 repeats, serial vs padded-batch).
  * variant 0 = base rubric; 1-3 = paraphrases. PARAPHRASES ARE UNVALIDATED DRAFTS: validate before quoting.
"""
import argparse, json, math, random, re, sys, time, zlib
from pathlib import Path

DATA = Path("data")
OUT = DATA / "judgments"
ROSTER = {  # 8-bit MLX conversions, pinned by revision. Same quantization for every judge.
    "qwen2.5-0.5b": ("mlx-community/Qwen2.5-0.5B-Instruct-8bit", "6474d4c1a05bd7bd634a9c993560e4e7272eda3d"),
    "llama3.2-1b": ("mlx-community/Llama-3.2-1B-Instruct-8bit", "d48cdf0a4ea22d893b7c63a99d6a693e24822795"),
    "llama3.2-3b": ("mlx-community/Llama-3.2-3B-Instruct-8bit", "ff054899609078569493def2823f9acd2780c0c9"),
    "qwen2.5-1.5b": ("mlx-community/Qwen2.5-1.5B-Instruct-8bit", "c7e1ff25efda4a2c8f7ee312380557fb42e145a2"),
    "qwen2.5-3b": ("mlx-community/Qwen2.5-3B-Instruct-8bit", "439033055dac10b43ab77228a2331b833af40a8b"),
    "qwen2.5-7b": ("mlx-community/Qwen2.5-7B-Instruct-8bit", "df11cb390bd674c24a9ddedf19f05bc09f6e0148"),
    "llama3.1-8b": ("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit", "142d428004044c37c441272c91316251d9aecc58"),
}
TEMPS, SEEDS = (0.3, 0.7, 1.0), range(10)
KINDS = ("faith", "rel", "prec", "rec", "corr", "abst")
ABSTAIN_FRAGMENT = "cannot answer this from the provided passages"
FMT = "\n\nAnswer with only one word: Yes or No."

TEMPLATES = {
    "faith": [
        "Passages:\n{passages}\n\nClaim: {claim}\n\nIs the claim fully supported by the passages above?",
        "Read the passages and then the claim.\n\nPassages:\n{passages}\n\nClaim: {claim}\n\nCan the claim be verified using only information in the passages?",
        "Passages:\n{passages}\n\nStatement: {claim}\n\nDoes the information in the passages entail this statement?",
        "Use only the passages below as evidence.\n\n{passages}\n\nClaim: {claim}\n\nIs the claim backed up by the passages?",
    ],
    "rel": [
        "Question: {question}\n\nAnswer: {answer}\n\nDoes the answer directly address the question that was asked? Ignore whether it is factually correct.",
        "Here is a question and a response.\n\nQuestion: {question}\n\nResponse: {answer}\n\nIs the response on topic and responsive to the question (correctness does not matter)?",
        "Question: {question}\n\nAnswer: {answer}\n\nIs this a relevant answer to the question, setting aside whether it is right or wrong?",
        "Decide whether the answer is relevant to the question.\n\nQuestion: {question}\nAnswer: {answer}\n\nIs the answer pertinent to what the question asks (not necessarily correct)?",
    ],
    "prec": [
        "Question: {question}\n\nReference answer: {reference}\n\nPassage: {passage}\n\nDoes the passage contain information that is useful for arriving at the reference answer?",
        "Question: {question}\nCorrect answer: {reference}\n\nPassage:\n{passage}\n\nIs this passage helpful for deriving the correct answer?",
        "Consider the question and its reference answer, then the passage.\n\nQuestion: {question}\nReference answer: {reference}\n\nPassage: {passage}\n\nWould this passage help someone produce the reference answer?",
        "Question: {question}\n\nGold answer: {reference}\n\nPassage: {passage}\n\nIs the passage relevant and useful for obtaining the gold answer?",
    ],
    "rec": [
        "Passages:\n{passages}\n\nSentence: {sentence}\n\nCan this sentence be attributed to the information in the passages?",
        "Using only these passages:\n{passages}\n\nSentence: {sentence}\n\nIs the information in the sentence present in the passages?",
        "Passages:\n{passages}\n\nTarget sentence: {sentence}\n\nIs the content of the target sentence supported by the passages?",
        "Check whether the passages contain the information in the sentence.\n\n{passages}\n\nSentence: {sentence}\n\nIs the sentence covered by the passages?",
    ],
    "corr": [
        "Question: {question}\n\nReference answer(s):\n{reference}\n\nCandidate answer: {answer}\n\nIs the candidate answer correct, i.e. consistent with the reference answer and not contradicting it?",
        "Question: {question}\n\nGold answer(s):\n{reference}\n\nResponse: {answer}\n\nDoes the response agree with the gold answer on the key facts?",
        "Compare a candidate answer with the reference.\n\nQuestion: {question}\nReference:\n{reference}\n\nCandidate: {answer}\n\nIs the candidate correct according to the reference?",
        "Question: {question}\n\nCorrect answer(s):\n{reference}\n\nProposed answer: {answer}\n\nWould the proposed answer be judged correct given the correct answer(s)?",
    ],
    "abst": [
        "Question: {question}\n\nAnswer: {answer}\n\nDoes the answer state that the question cannot be answered or decline to give an answer?",
        "Question: {question}\n\nResponse: {answer}\n\nIs the response a refusal or an admission that it cannot answer?",
        "Question: {question}\n\nAnswer: {answer}\n\nDoes the answer say it lacks the information needed to answer?",
        "Question: {question}\nAnswer: {answer}\n\nIs this an abstention (the answerer declines or says it cannot answer)?",
    ],
}


# ---------- units ----------
def split_sents(t):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(`])|\n+", t.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def build_units(ev, a):
    """All judgment units for one item -> [(kind, unit_idx, template_fields)]."""
    ctx, q, ans = a["contexts"], a["question"], a["answer"]
    P = "\n\n".join(f"[{i}] {c['title']}: {c['text']}" for i, c in enumerate(ctx, 1))
    U = [("faith", i, dict(passages=P, claim=s)) for i, s in
         enumerate(s for s in split_sents(ans) if ABSTAIN_FRAGMENT not in s.lower())]
    U += [("rel", 0, dict(question=q, answer=ans)), ("abst", 0, dict(question=q, answer=ans))]
    if ev["answerable"]:  # reference-based units exist only when a gold answer exists
        ref = ev["gold_answers"][0]
        U += [("prec", i, dict(question=q, reference=ref, passage=f"{c['title']}: {c['text']}"))
              for i, c in enumerate(ctx)]
        U += [("rec", i, dict(passages=P, sentence=s)) for i, s in enumerate(split_sents(ref))]
        U.append(("corr", 0, dict(question=q, answer=ans,
                                  reference="\n".join(f"- {g}" for g in ev["gold_answers"][:3]))))
    return U


def sample(judge, key, d, T, seed):
    u = random.Random(zlib.crc32(f"{judge}|{key}|{T}|{seed}".encode())).random()
    return int(u < 1 / (1 + math.exp(-max(-50.0, min(50.0, d / T)))))


# ---------- model ----------
class Judge:
    def __init__(self, name):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        repo, rev = ROSTER[name]
        self.mx, self.name = mx, name
        mx.set_cache_limit(512 * 1024 * 1024)  # RAM is tight (16 GB shared); the OS was killing us (Killed: 9)
        self.model, self.tok = load(snapshot_download(
            repo, revision=rev, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "tokenizer*"]))
        first = lambda w: self.tok.encode(w, add_special_tokens=False)[0]
        self.yes, self.no = first("Yes"), first("No")
        assert self.yes != self.no

    def encode(self, kind, variant, fields):
        msg = [{"role": "user", "content": TEMPLATES[kind][variant].format(**fields) + FMT}]
        text = self.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return self.tok.encode(text, add_special_tokens=False)  # template already carries BOS if any

    def forward(self, seqs):
        """[(d, mass)] per sequence. Right-padded batch + causal mask => padding cannot affect real tokens."""
        mx, m = self.mx, self.model
        L = max(map(len, seqs))
        h = m.model(mx.array([s + [0] * (L - len(s)) for s in seqs]))
        h = mx.stack([h[k, len(s) - 1] for k, s in enumerate(seqs)])
        lg = (m.lm_head(h) if hasattr(m, "lm_head") else m.model.embed_tokens.as_linear(h)).astype(mx.float32)
        d = lg[:, self.yes] - lg[:, self.no]
        mass = mx.exp(mx.logsumexp(lg[:, [self.yes, self.no]], axis=-1) - mx.logsumexp(lg, axis=-1))
        mx.eval(d, mass)
        return list(zip(d.tolist(), mass.tolist()))


# ---------- run ----------
def load_items(n):
    ev = {x["id"]: x for x in map(json.loads, (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines())}
    ans = [json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    return ev, ans[:n] if n else ans


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.judge}{'_' + args.tag if args.tag else ''}.jsonl"
    key = lambda r: (r["id"], r["kind"], r["unit"], r["variant"], r["cond"], r["rep"])
    done = {key(json.loads(l)) for l in path.read_text().splitlines()} if path.exists() else set()
    ev, ans = load_items(args.n)
    J = Judge(args.judge)
    variants = [int(v) for v in args.variants.split(",")]

    def phase(items, variant, cond, rep, bs):
        jobs = [dict(judge=args.judge, id=a["id"], kind=k, unit=u, variant=variant, cond=cond, rep=rep, f=f)
                for a in items for k, u, f in build_units(ev[a["id"]], a)]
        jobs = [j for j in jobs if key(j) not in done]
        random.Random(zlib.crc32(f"{variant}{cond}{rep}".encode())).shuffle(jobs)  # order differs per phase
        t0 = time.time()
        for s in range(0, len(jobs), 16):
            chunk, t1 = jobs[s:s + 16], time.time()
            res = []
            for b in range(0, len(chunk), bs):
                part = chunk[b:b + bs]
                res += J.forward([J.encode(j["kind"], variant, j["f"]) for j in part])
            sec = (time.time() - t1) / len(chunk)
            with open(path, "a", encoding="utf-8") as fh:
                for j, (d, mass) in zip(chunk, res):
                    r = {k: v for k, v in j.items() if k != "f"}
                    r.update(d=d, mass=mass, y0=int(d > 0), sec=round(sec, 3))
                    if cond == "serial" and rep == 0:
                        k = "|".join(map(str, key(j)))
                        r["samples"] = {str(T): [sample(args.judge, k, d, T, sd) for sd in SEEDS] for T in TEMPS}
                    fh.write(json.dumps(r) + "\n")
            J.mx.clear_cache()
            print(f"v{variant} {cond} rep{rep}: {s + len(chunk)}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    for v in variants:
        phase(ans, v, "serial", 0, 1)
    if args.det_n:  # T=0 determinism: same prompts, different order, serial vs padded batch of 4
        sub = ans[:args.det_n]
        phase(sub, 0, "serial", 1, 1); phase(sub, 0, "serial", 2, 1); phase(sub, 0, "batch4", 0, 4)
    print("done:", path)


def summary(path):
    R = [json.loads(l) for l in open(path)]
    main = [r for r in R if r["cond"] == "serial" and r["rep"] == 0]
    med = lambda x: sorted(x)[len(x) // 2]
    print(f"{len(R)} records; median sec/forward = {med([r['sec'] for r in R]):.2f}")
    for k in KINDS:
        rows = [r for r in main if r["kind"] == k and r["variant"] == 0]
        if rows:
            print(f"{k:6} n={len(rows):4} yes-rate={sum(r['y0'] for r in rows) / len(rows):.2f} "
                  f"mass median={med([r['mass'] for r in rows]):.3f} min={min(r['mass'] for r in rows):.3f}")
    key = lambda r: (r["id"], r["kind"], r["unit"], r["variant"])
    base = {key(r): r for r in main}
    groups = {}
    for r in R:
        if not (r["cond"] == "serial" and r["rep"] == 0):
            groups.setdefault((r["cond"], r["rep"], r["variant"]), []).append(r)
    for (cond, rep, v), rows in sorted(groups.items()):
        if v == 0:
            dd = [abs(r["d"] - base[key(r)]["d"]) for r in rows if key(r) in base]
            fl = sum(r["y0"] != base[key(r)]["y0"] for r in rows if key(r) in base)
            print(f"vs base: {cond} rep{rep}: n={len(dd)} max|dd|={max(dd):.5f} mean|dd|={sum(dd) / len(dd):.6f} label flips={fl}")


def selftest():
    assert split_sents("A b c. D e f!\nG h i") == ["A b c.", "D e f!", "G h i"]
    ev = dict(answerable=True, gold_answers=["Alpha one. Beta two.", "x"])
    a = dict(question="q?", answer=f"I {ABSTAIN_FRAGMENT}. But maybe so.", contexts=[dict(title="T", text="t")] * 3)
    U = build_units(ev, a)
    cnt = {k: sum(u[0] == k for u in U) for k in KINDS}
    assert cnt == dict(faith=1, rel=1, prec=3, rec=2, corr=1, abst=1), cnt  # abstain sentence not a claim
    assert {u[0] for u in build_units(dict(answerable=False, gold_answers=[]), a)} == {"faith", "rel", "abst"}
    for kind, f in {u[0]: u[2] for u in U}.items():  # every variant formats cleanly
        for v in range(4):
            TEMPLATES[kind][v].format(**f)
    assert all(len(TEMPLATES[k]) == 4 for k in KINDS)
    assert all(sample("j", "k", 100, 0.3, s) == 1 and sample("j", "k", -100, 0.3, s) == 0 for s in SEEDS)
    assert sample("j", "k", 0.7, 1.0, 3) == sample("j", "k", 0.7, 1.0, 3)  # reproducible
    rate = sum(sample("j", "k", 1.0, 1.0, s) for s in range(4000)) / 4000
    assert abs(rate - 1 / (1 + math.exp(-1.0))) < 0.03, rate  # matches sigmoid
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run", "summary"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--judge", choices=list(ROSTER))
    ap.add_argument("--variants", default="0,1,2,3")
    ap.add_argument("--n", type=int, default=0, help="only first N items (0 = all)")
    ap.add_argument("--det-n", type=int, default=100, help="items in the T=0 determinism repeats (0 = skip)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest()
    elif a.cmd == "summary": summary(a.path)
    else:
        assert a.judge, "--judge required"
        run(a)
