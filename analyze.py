#!/usr/bin/env python3
"""Statistics + figures for the study.

    python3 analyze.py selftest
    python3 analyze.py run [--suffix _pilot] [--boots 2000]

Reads data/judgments/<judge><suffix>.jsonl and data/refmetrics.jsonl; writes data/results/*.csv, results.json
and figures/*.png. Sections that need data that is not there yet (e.g. paraphrase variants) are skipped.

Pre-declared choices: primary sampling temperature T=0.3 ("low"), 0.7 and 1.0 are sensitivity; all CIs are
item-cluster bootstraps; Holm correction over the family of capacity-trend tests.
"""
import argparse, itertools, json, math, sys
from pathlib import Path
import numpy as np

import judge as J

DATA, RES, FIG = Path("data"), Path("data/results"), Path("figures")
# family, parameters in billions (from the model cards; includes embeddings)
META = {"qwen2.5-0.5b": ("Qwen2.5", 0.494), "qwen2.5-1.5b": ("Qwen2.5", 1.54), "qwen2.5-3b": ("Qwen2.5", 3.09),
        "qwen2.5-7b": ("Qwen2.5", 7.62), "llama3.2-1b": ("Llama", 1.24), "llama3.2-3b": ("Llama", 3.21),
        "llama3.1-8b": ("Llama", 8.03)}
KINDS, TEMPS, NS = J.KINDS, J.TEMPS, len(J.SEEDS)
PT = 0.3  # primary temperature
KI = {k: i for i, k in enumerate(KINDS)}
P = 1 + len(TEMPS) * NS  # pass 0 = T0 verdict, then (T, seed)


# ---------- statistics helpers (covered by selftest) ----------
def kalpha_nominal(M):
    """Krippendorff alpha, nominal, complete data. M: (units, raters). NaN if no variation is possible."""
    M = np.asarray(M)
    n_u, m = M.shape
    cats = np.unique(M)
    o = np.zeros((len(cats), len(cats)))
    for row in M:
        c = np.array([(row == k).sum() for k in cats], float)
        o += (np.outer(c, c) - np.diag(c)) / (m - 1)
    n_c, n = o.sum(1), o.sum()
    de = n * n - (n_c ** 2).sum()
    return np.nan if de == 0 else 1 - (n - 1) * (o.sum() - np.trace(o)) / de


def boot_ci(v, rng, B):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan,) * 3
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return v.mean(), lo, hi


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def slope(x, Y):
    """OLS slope of each column of Y (judges x B) on x."""
    xc = x - x.mean()
    return (xc[:, None] * (Y - Y.mean(0))).sum(0) / (xc ** 2).sum()


def varcomp2(X):
    """Random-effects variance components, items x judges x replicates (seeds). -> dict of shares."""
    a, b, n = X.shape
    g = X.mean(); mi, mj, mij = X.mean((1, 2)), X.mean((0, 2)), X.mean(2)
    MSA = b * n * ((mi - g) ** 2).sum() / (a - 1)
    MSB = a * n * ((mj - g) ** 2).sum() / (b - 1)
    MSAB = n * ((mij - mi[:, None] - mj[None] + g) ** 2).sum() / ((a - 1) * (b - 1))
    MSE = ((X - mij[:, :, None]) ** 2).sum() / (a * b * (n - 1))
    c = dict(item=(MSA - MSAB) / (b * n), judge=(MSB - MSAB) / (a * n), item_x_judge=(MSAB - MSE) / n, seed_noise=MSE)
    return _shares(c)


def varcomp3(X):
    """Random-effects components, items x judges x variants (one obs per cell)."""
    I, Jn, K = X.shape
    g = X.mean(); mi, mj, mk = X.mean((1, 2)), X.mean((0, 2)), X.mean((0, 1))
    mij, mik, mjk = X.mean(2), X.mean(1), X.mean(0)
    MSI = J_K = Jn * K * ((mi - g) ** 2).sum() / (I - 1)
    MSJ = I * K * ((mj - g) ** 2).sum() / (Jn - 1)
    MSK = I * Jn * ((mk - g) ** 2).sum() / (K - 1)
    MSIJ = K * ((mij - mi[:, None] - mj[None] + g) ** 2).sum() / ((I - 1) * (Jn - 1))
    MSIK = Jn * ((mik - mi[:, None] - mk[None] + g) ** 2).sum() / ((I - 1) * (K - 1))
    MSJK = I * ((mjk - mj[:, None] - mk[None] + g) ** 2).sum() / ((Jn - 1) * (K - 1))
    r = X - mij[:, :, None] - mik[:, None, :] - mjk[None] + mi[:, None, None] + mj[None, :, None] + mk[None, None, :] - g
    MSE = (r ** 2).sum() / ((I - 1) * (Jn - 1) * (K - 1))
    c = dict(item=(MSI - MSIJ - MSIK + MSE) / (Jn * K), judge=(MSJ - MSIJ - MSJK + MSE) / (I * K),
             variant=(MSK - MSIK - MSJK + MSE) / (I * Jn), item_x_judge=(MSIJ - MSE) / K,
             item_x_variant=(MSIK - MSE) / Jn, judge_x_variant=(MSJK - MSE) / I, residual=MSE)
    return _shares(c)


def _shares(c):
    c = {k: max(v, 0.0) for k, v in c.items()}; t = sum(c.values()) or 1.0
    return {k: v / t for k, v in c.items()}


def score(kind, L):
    """Item score for one metric from unit labels L (units x passes)."""
    if kind in ("faith", "rec"):
        return L.mean(0)
    if kind == "prec":  # rank-weighted precision over the ordered retrieved passages
        pk = np.cumsum(L, 0) / np.arange(1, len(L) + 1)[:, None]
        return np.where(L.sum(0) > 0, (pk * L).sum(0) / np.maximum(L.sum(0), 1), 0.0)
    return L[0]  # rel / corr / abst: single unit


# ---------- data ----------
def load_all(suffix):
    ev = {x["id"]: x for x in map(json.loads, (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines())}
    ans = [json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    exp = {}
    for a in ans:
        for k, u, _ in J.build_units(ev[a["id"]], a):
            exp[(a["id"], k)] = exp.get((a["id"], k), 0) + 1
    recs = {}
    for j in META:
        p = J.OUT / f"{j}{suffix}.jsonl"
        if p.exists():
            r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            if any(x["variant"] == 0 for x in r):
                recs[j] = r
    ref = {r["id"]: r for r in map(json.loads, (DATA / "refmetrics.jsonl").read_text().splitlines())}
    return ev, ans, exp, recs, ref


def tensor(recs, variant, ans, exp):
    """(items, metrics, passes) item scores for one judge+variant; NaN where any unit is missing."""
    by = {}
    for r in recs:
        if r["variant"] == variant and r["cond"] == "serial" and r["rep"] == 0:
            by.setdefault((r["id"], r["kind"]), {})[r["unit"]] = \
                [r["y0"]] + [r["samples"][str(T)][s] for T in TEMPS for s in range(NS)]
    X = np.full((len(ans), len(KINDS), P), np.nan)
    for i, a in enumerate(ans):
        for k in KINDS:
            n, have = exp.get((a["id"], k), 0), by.get((a["id"], k), {})
            if n and len(have) == n:
                X[i, KI[k]] = score(k, np.array([have[u] for u in range(n)], float))
    return X


def block(X, T):
    t = TEMPS.index(T)
    return X[:, :, 1 + t * NS: 1 + (t + 1) * NS]


def selftest():
    assert abs(kalpha_nominal(np.array([[0, 0], [1, 1], [0, 1]])) - kalpha_nominal(np.array([[0, 0], [1, 1], [0, 1]]))) == 0
    assert kalpha_nominal(np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]])) == 1.0
    assert np.isnan(kalpha_nominal(np.ones((4, 3), int)))
    assert kalpha_nominal(np.array([[0, 1], [1, 0], [0, 1], [1, 0]])) < 0  # systematic disagreement
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    x = np.array([0., 1, 2, 3]); assert np.allclose(slope(x, np.array([[1.], [3], [5], [7]])), [2.0])
    rng = np.random.default_rng(0)
    item = rng.normal(0, 2, (400, 1, 1))
    judge = np.array([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]).reshape(1, 6, 1)   # fixed effects, var 1.26 (random draws of 6 are too noisy to test)
    vi, vj = item.var(ddof=1), 1.26
    s = varcomp2(item + judge + rng.normal(0, 1, (400, 6, 10)))
    tot = vi + vj + 1.0
    assert abs(s["item"] - vi / tot) < 0.05 and abs(s["judge"] - vj / tot) < 0.05 and abs(s["seed_noise"] - 1 / tot) < 0.03, s
    var = np.array([0.9, 0.3, -0.3, -0.9]).reshape(1, 1, 4)                   # variant effect, var 0.6
    s3 = varcomp3(item + judge + var + rng.normal(0, 0.5, (400, 6, 4)))
    t3 = vi + vj + 0.6 + 0.25
    assert abs(s3["item"] - vi / t3) < 0.06 and abs(s3["variant"] - 0.6 / t3) < 0.06 and abs(s3["residual"] - 0.25 / t3) < 0.03, s3
    L = np.array([[1], [0], [1]])  # AP = (1/1 + 2/3) / 2
    assert abs(score("prec", L)[0] - (1 + 2 / 3) / 2) < 1e-9 and score("prec", np.zeros((3, 1)))[0] == 0
    m, lo, hi = boot_ci(np.array([0, 1] * 50), np.random.default_rng(1), 500)
    assert 0.35 < lo < m < hi < 0.65
    print("selftest ok")


# ---------- analysis ----------
def run(args):
    import pandas as pd
    from scipy import stats
    from scipy.cluster import hierarchy
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    ev, ans, exp, recs, ref = load_all(args.suffix)
    B, rng = args.boots, np.random.default_rng(1234)
    judges = [j for j in META if j in recs]
    fam = {j: META[j][0] for j in judges}
    X = {j: tensor(recs[j], 0, ans, exp) for j in judges}
    XV = {j: {v: tensor(recs[j], v, ans, exp) for v in range(4)} for j in judges}
    have_para = {j: all(np.isfinite(XV[j][v][:, KI["abst"], 0]).any() for v in range(4)) for j in judges}
    ids = [a["id"] for a in ans]
    out = {"judges": judges, "boots": B, "primary_T": PT}
    print("judges with data:", judges)

    # Table 1: roster
    pd.DataFrame([dict(judge=j, family=META[j][0], params_B=META[j][1], hf_repo=J.ROSTER[j][0], revision=J.ROSTER[j][1],
                       quant="8-bit MLX") for j in META]).to_csv(RES / "table1_roster.csv", index=False)

    # descriptives
    rows = []
    for j in judges:
        for k in KINDS:
            v = X[j][:, KI[k], 0]; v = v[~np.isnan(v)]
            rows.append(dict(judge=j, metric=k, n_items=len(v), mean_T0=v.mean() if len(v) else np.nan,
                             constant=bool(len(v) and v.std() == 0)))
        mass = [r["mass"] for r in recs[j]]
        out.setdefault("min_yes_no_mass", {})[j] = float(min(mass))
    pd.DataFrame(rows).to_csv(RES / "descriptives.csv", index=False)

    # RQ1a: within-judge noise floor (SD across seeds) per judge x metric x T
    rows, noise = [], {}
    for j in judges:
        for T in TEMPS:
            blk = block(X[j], T)
            for k in KINDS:
                sd = blk[:, KI[k]].std(1, ddof=1)
                m, lo, hi = boot_ci(sd, rng, B)
                mean_score = np.nanmean(X[j][:, KI[k], 0])
                rows.append(dict(judge=j, T=T, metric=k, mean_item_SD=m, ci_lo=lo, ci_hi=hi, ci_width=hi - lo,
                                 CV=m / mean_score if mean_score else np.nan,
                                 frac_items_with_variation=float(np.nanmean(sd > 0)) if np.isfinite(sd).any() else np.nan))
                if T == PT:
                    noise[(j, k)] = sd
    pd.DataFrame(rows).to_csv(RES / "noise_floor.csv", index=False)

    # RQ1b: T=0 determinism (serial x3 + padded batch of 4, same prompts, shuffled order)
    rows = []
    for j in judges:
        combos = [("serial", 0), ("serial", 1), ("serial", 2), ("batch4", 0)]
        d = {c: {(r["id"], r["kind"], r["unit"]): r for r in recs[j] if r["variant"] == 0 and (r["cond"], r["rep"]) == c}
             for c in combos}
        keys = set.intersection(*[set(v) for v in d.values()])
        if not keys:
            continue
        keys = sorted(keys)
        D = np.array([[d[c][k]["d"] for c in combos] for k in keys]); Y = (D > 0).astype(int)
        rows.append(dict(judge=j, n_units=len(keys), exact_match_all_4=float((D.max(1) == D.min(1)).mean()),
                         max_abs_dd=float(np.abs(D - D[:, :1]).max()), label_agreement=float((Y.min(1) == Y.max(1)).mean()),
                         krippendorff_alpha_T0=kalpha_nominal(Y)))
    det = pd.DataFrame(rows); det.to_csv(RES / "table2_determinism.csv", index=False)
    print("\nTable 2 (T=0 determinism):\n", det.to_string(index=False) if len(det) else "no repeats yet")

    # verdict flips + inter-judge agreement (correctness verdict on answerable items; abstention on all)
    flips, agree_vec, flip_vec = [], {}, {}
    for kind in ("corr", "abst"):
        k = KI[kind]
        V = {j: X[j][:, k, 0] for j in judges}
        for j in judges:
            others = [o for o in judges if o != j]
            if others:
                agree_vec[(kind, j)] = np.nanmean(np.stack([(V[j] == V[o]).astype(float) for o in others]), 0)
                agree_vec[(kind, j)][np.isnan(V[j])] = np.nan
            for T in TEMPS:
                blk = block(X[j], T)[:, k]
                fv = np.where(np.isnan(blk[:, 0]), np.nan, (blk.min(1) != blk.max(1)).astype(float))
                flip_vec[(kind, j, T)] = fv
                m, lo, hi = boot_ci(fv, rng, B)
                flips.append(dict(kind=kind, source="seeds", judge=j, T=T, flip_rate=m, ci_lo=lo, ci_hi=hi))
            if have_para[j]:
                Z = np.stack([XV[j][v][:, k, 0] for v in range(4)], 1)
                fv = np.where(np.isnan(Z).any(1), np.nan, (Z.min(1) != Z.max(1)).astype(float))
                flip_vec[(kind, j, "para")] = fv
                m, lo, hi = boot_ci(fv, rng, B)
                flips.append(dict(kind=kind, source="paraphrase", judge=j, T=0, flip_rate=m, ci_lo=lo, ci_hi=hi))
        for name, grp in (("all_judges", judges), ("qwen_ladder", [j for j in judges if fam[j] == "Qwen2.5"]),
                          ("llama_ladder", [j for j in judges if fam[j] == "Llama"])):
            if len(grp) >= 2:
                Z = np.stack([V[j] for j in grp], 1); ok = ~np.isnan(Z).any(1)
                fv = np.where(ok, (Z.min(1) != Z.max(1)).astype(float), np.nan)
                m, lo, hi = boot_ci(fv, rng, B)
                al = kalpha_nominal(Z[ok].astype(int)) if ok.sum() > 1 else np.nan
                flips.append(dict(kind=kind, source=f"judges:{name}", judge="-", T=0, flip_rate=m, ci_lo=lo, ci_hi=hi,
                                  krippendorff_alpha=al, n_items=int(ok.sum())))
        if len(judges) > 1:  # pairwise agreement matrix
            M = pd.DataFrame([[(V[a] == V[b])[~(np.isnan(V[a]) | np.isnan(V[b]))].mean()
                               for b in judges] for a in judges], index=judges, columns=judges)
            M.to_csv(RES / f"pairwise_agreement_{kind}.csv")
    fl = pd.DataFrame(flips); fl.to_csv(RES / "verdict_flips.csv", index=False)
    print("\nVerdict flips (headline):\n", fl[fl.kind == "corr"].round(3).to_string(index=False))

    # paraphrase sensitivity + variance decomposition
    rows = []
    for j in judges:
        if not have_para[j]:
            continue
        for k in KINDS:
            Z = np.stack([XV[j][v][:, KI[k], 0] for v in range(4)], 1)
            sd = Z.std(1, ddof=1); m, lo, hi = boot_ci(sd, rng, B)
            rows.append(dict(judge=j, metric=k, mean_item_SD_across_paraphrases=m, ci_lo=lo, ci_hi=hi,
                             mean_item_SD_across_seeds_T03=boot_ci(noise[(j, k)], rng, B)[0]))
    if rows:
        pd.DataFrame(rows).to_csv(RES / "paraphrase_sensitivity.csv", index=False)
    vc = []
    for k in KINDS:
        for T in TEMPS:
            if len(judges) < 2:
                break
            A = np.stack([block(X[j], T)[:, KI[k]] for j in judges], 1)  # items x judges x seeds
            ok = ~np.isnan(A).any((1, 2))
            if ok.sum() > 5:
                vc.append(dict(design="item x judge x seed", metric=k, T=T, n_items=int(ok.sum()), **varcomp2(A[ok])))
        if all(have_para.values()) and len(judges) > 1:
            A = np.stack([np.stack([XV[j][v][:, KI[k], 0] for v in range(4)], 1) for j in judges], 1)
            ok = ~np.isnan(A).any((1, 2))
            if ok.sum() > 5:
                vc.append(dict(design="item x judge x paraphrase (T=0)", metric=k, T=0, n_items=int(ok.sum()), **varcomp3(A[ok])))
    if vc:
        pd.DataFrame(vc).to_csv(RES / "variance_components.csv", index=False)

    # RQ2: capacity trend (item-cluster bootstrap of the slope on log2 params; Holm over the whole family)
    outcomes = {}
    for k in KINDS:
        outcomes[f"noise_SD_{k}"] = {j: noise[(j, k)] for j in judges}
    for kind in ("corr", "abst"):
        outcomes[f"seed_flip_{kind}_T{PT}"] = {j: flip_vec[(kind, j, PT)] for j in judges}
        outcomes[f"interjudge_agreement_{kind}"] = {j: agree_vec[(kind, j)] for j in judges if (kind, j) in agree_vec}
    rows = []
    for name, vec in outcomes.items():
        for scope, grp in (("all", judges), ("qwen_only", [j for j in judges if fam[j] == "Qwen2.5"])):
            grp = [j for j in grp if j in vec]
            if len(grp) < 3:
                continue
            M = np.stack([vec[j] for j in grp]); ok = ~np.isnan(M).any(0); M = M[:, ok]
            x = np.log2(np.array([META[j][1] for j in grp]))
            idx = rng.integers(0, M.shape[1], (B, M.shape[1]))
            Yb = np.stack([M[:, ix].mean(1) for ix in idx], 1)  # judges x B
            sb = slope(x, Yb); s0 = slope(x, M.mean(1)[:, None])[0]
            pb = (min((sb >= 0).sum(), (sb <= 0).sum()) + 1) / (B + 1) * 2
            lr = stats.linregress(x, M.mean(1))
            rows.append(dict(outcome=name, scope=scope, n_judges=len(grp), slope_per_doubling=s0,
                             ci_lo=np.percentile(sb, 2.5), ci_hi=np.percentile(sb, 97.5), p_bootstrap=min(pb, 1.0), p_ols=lr.pvalue))
    cap = pd.DataFrame(rows)
    if len(cap):
        # Judge-level OLS p is the inferential one (the item bootstrap holds the judges fixed, so it only
        # reflects item-sampling noise and would overstate significance with 4-7 judges). Holm over all tests.
        cap["p_holm"] = holm(cap.p_ols.values)
        cap.to_csv(RES / "capacity_trend.csv", index=False)
        print("\nCapacity trend (slope per doubling of parameters):\n", cap.round(4).to_string(index=False))

    # RQ3: cross-metric agreement (T0 scores, base rubric) + redundancy
    names = list(KINDS) + ["rougeL", "bertscore", "hit@3"]
    R = {n: np.array([np.nan if ref[i][n] is None else ref[i][n] for i in ids], float) for n in ("rougeL", "bertscore", "hit@3")}
    red = []
    for j in judges:
        C = np.column_stack([X[j][:, KI[k], 0] for k in KINDS] + [R[n] for n in ("rougeL", "bertscore", "hit@3")])
        rho = np.full((len(names), len(names)), np.nan)
        for a, b in itertools.product(range(len(names)), repeat=2):
            m = ~np.isnan(C[:, a]) & ~np.isnan(C[:, b])
            if m.sum() > 5 and C[m, a].std() > 0 and C[m, b].std() > 0:
                rho[a, b] = stats.spearmanr(C[m, a], C[m, b])[0]
        pd.DataFrame(rho, index=names, columns=names).to_csv(RES / f"crossmetric_{j}.csv")
        live = [i for i, k in enumerate(KINDS) if np.isfinite(rho[i, i])]  # judge metrics that actually vary
        if len(live) >= 2:
            S = np.nan_to_num(rho[np.ix_(live, live)]); np.fill_diagonal(S, 1.0)
            lam = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
            cl = hierarchy.fcluster(hierarchy.linkage(
                1 - np.abs(S)[np.triu_indices(len(live), 1)], "average"), 0.5, "distance")
            red.append(dict(judge=j, live_metrics=len(live), constant_metrics=len(KINDS) - len(live),
                            pca_components_for_90pct=int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.9) + 1),
                            participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()), clusters_abs_rho_ge_0_5=int(len(set(cl)))))
    if red:
        pd.DataFrame(red).to_csv(RES / "redundancy.csv", index=False)
        print("\nRedundancy:\n", pd.DataFrame(red).round(2).to_string(index=False))

    # validity: abstention vs rule label; correctness vs ROUGE-L quartile extremes
    rule = np.array([float(ABSTAIN in a["answer"].lower()) for a in ans])
    q = np.nanpercentile(R["rougeL"], [25, 75]); rl = R["rougeL"]
    rows = []
    for j in judges:
        for v in range(4):
            if not np.isfinite(XV[j][v][:, KI["abst"], 0]).any():
                continue
            y = XV[j][v][:, KI["abst"], 0]; ok = ~np.isnan(y); tp = ((y == 1) & (rule == 1) & ok).sum()
            fp, fn = ((y == 1) & (rule == 0) & ok).sum(), ((y == 0) & (rule == 1) & ok).sum()
            c = XV[j][v][:, KI["corr"], 0]
            pos, neg = (rl >= q[1]) & ~np.isnan(c), (rl <= q[0]) & ~np.isnan(c)
            bal = np.nan if pos.sum() == 0 or neg.sum() == 0 else 0.5 * ((c[pos] == 1).mean() + (c[neg] == 0).mean())
            rows.append(dict(judge=j, variant=v, abst_accuracy=float((y[ok] == rule[ok]).mean()),
                             abst_precision=tp / max(tp + fp, 1), abst_recall=tp / max(tp + fn, 1),
                             corr_balanced_acc_vs_rougeL_extremes=bal))
    if rows:
        pd.DataFrame(rows).to_csv(RES / "validity.csv", index=False)

    json.dump(out, open(RES / "results.json", "w"), indent=1, default=float)
    make_figures(judges, fam, noise, fl, cap, X, RES, vc)
    print("\nwrote", RES, "and", FIG)


ABSTAIN = J.ABSTAIN_FRAGMENT


# ---------- figures ----------
def make_figures(judges, fam, noise, fl, cap, X, RES, vc):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    FC = {"Qwen2.5": "#2a78d6", "Llama": "#eb6834"}     # categorical slots 1-2 (validated palette)
    MK = {"Qwen2.5": "o", "Llama": "s"}
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c9c8c3",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
                         "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "legend.frameon": False})
    ticks = [0.5, 1, 2, 4, 8]

    def xaxis(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}B" for t in ticks])
        ax.minorticks_off()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("judge parameters (log scale)")

    def draw(ax, df, ycol, lo=None, hi=None, **kw):
        for f in FC:
            sub = df[df.judge.map(fam) == f].copy(); sub["x"] = sub.judge.map(lambda j: META[j][1]); sub = sub.sort_values("x")
            if sub.empty:
                continue
            ax.plot(sub.x, sub[ycol], color=FC[f], marker=MK[f], lw=1.8, ms=6, mec=SURF, mew=1.2, label=f, **kw)
            if lo:
                ax.vlines(sub.x, sub[lo], sub[hi], color=FC[f], lw=1.2)

    nf = pd.read_csv(RES / "noise_floor.csv"); nf = nf[nf["T"] == PT]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True)
    for ax, k in zip(axs.flat, KINDS):
        draw(ax, nf[nf.metric == k], "mean_item_SD", "ci_lo", "ci_hi"); xaxis(ax); ax.set_title(k); ax.set_ylim(bottom=0)
    axs[0, 0].set_ylabel("mean within-item SD"); axs[1, 0].set_ylabel("mean within-item SD"); axs[0, 0].legend(title="family")
    fig.suptitle(f"Figure 1. Seed noise floor vs judge size (T={PT}, 95% item-bootstrap CI)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig1_noise_vs_size.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    pa = pd.read_csv(RES / "pairwise_agreement_corr.csv", index_col=0) if (RES / "pairwise_agreement_corr.csv").exists() else None
    if pa is not None:
        ag = pd.DataFrame({"judge": pa.index, "agree": [(pa.loc[j].drop(j)).mean() for j in pa.index]})
        draw(axs[0], ag, "agree"); xaxis(axs[0]); axs[0].set_ylabel("mean pairwise agreement with other judges")
        axs[0].set_title("Inter-judge agreement (correctness verdict, T=0)"); axs[0].legend(title="family")
    for src, ls, lab in (("seeds", "-", f"across seeds (T={PT})"), ("paraphrase", "--", "across wordings (T=0)")):
        sub = fl[(fl.kind == "corr") & (fl.source == src) & ((fl["T"] == PT) | (src == "paraphrase"))]
        for f in FC:
            s = sub[sub.judge.map(lambda j: fam.get(j)) == f].copy()
            if s.empty:
                continue
            s["x"] = s.judge.map(lambda j: META[j][1]); s = s.sort_values("x")
            axs[1].plot(s.x, s.flip_rate, color=FC[f], marker=MK[f], ls=ls, lw=1.8, ms=6, mec=SURF, label=f"{f}, {lab}")
    xaxis(axs[1]); axs[1].set_ylabel("verdict-flip rate"); axs[1].set_ylim(bottom=0)
    axs[1].set_title("Correctness-verdict flips"); axs[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Figure 2. Agreement and flips vs judge size", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig2_agreement_flips_vs_size.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bgr", ["#256abf", "#f0efec", "#d03b3b"])  # blue - gray - red, gray at 0
    show = [j for j in ("qwen2.5-1.5b", "qwen2.5-7b", "llama3.1-8b") if (RES / f"crossmetric_{j}.csv").exists()]
    if show:
        fig, axs = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.6), squeeze=False)
        for ax, j in zip(axs[0], show):
            C = pd.read_csv(RES / f"crossmetric_{j}.csv", index_col=0)
            ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1); ax.grid(False)
            ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=60, ha="right"); ax.set_yticks(range(len(C)))
            ax.set_yticklabels(C.index); ax.set_title(j)
            for (a, b), v in np.ndenumerate(C.values):
                ax.text(b, a, "" if np.isnan(v) else f"{v:.1f}", ha="center", va="center", fontsize=6, color=INK)
        fig.suptitle("Figure 3. Spearman correlation between metrics (blank = constant metric)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig3_crossmetric_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    seed_vc = [r for r in vc if r["design"].startswith("item x judge x seed") and r["T"] == 0.7]
    if seed_vc:
        parts, col = ["item", "judge", "item_x_judge", "seed_noise"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        fig, ax = plt.subplots(figsize=(7.5, 3.6)); left = np.zeros(len(seed_vc))
        for p, c in zip(parts, col):
            v = np.array([r[p] for r in seed_vc]); ax.barh(range(len(seed_vc)), v, left=left, color=c, label=p.replace("_", " "),
                                                             edgecolor=SURF, linewidth=2); left += v
        ax.set_yticks(range(len(seed_vc))); ax.set_yticklabels([r["metric"] for r in seed_vc]); ax.invert_yaxis()
        ax.set_xlabel("share of variance (T=0.7)"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
        ax.grid(axis="y", visible=False)
        fig.suptitle("Figure 5. Where the variance comes from", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout(); fig.savefig(FIG / "fig5_variance_components.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--boots", type=int, default=2000)
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else run(a)
