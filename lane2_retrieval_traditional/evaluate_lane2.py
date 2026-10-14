"""
evaluate_lane2.py — scores Lane 2's retrieve() with Lane 1's shared evaluation harness.

    python evaluate_lane2.py

What it does
  1. Dev split: small sweep of BM25 settings (preprocessing x k1 x b) to check the defaults
     are reasonable  -> results/dev_tuning_sweep.csv
  2. Test split (claims with a gold_evidence_id): Recall@1/3/5/10 and MRR with 95% bootstrap CIs for
       - Lane 2 BM25   (this lane's retrieve(), the submitted traditional baseline)
       - Lane 2 TF-IDF (retrieve(method="tfidf"))
       - Lane 1 naive BM25 (whitespace tokens, no preprocessing) as a reference point
     plus paired bootstrap significance tests          -> results/lane2_results.md / .csv
  3. Per-claim rank of the gold evidence (for Lane 1's error analysis) -> results/per_claim_test.csv
  4. Bar chart with CI error bars                       -> results/lane2_retrieval_ci.png
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LANE1 = HERE.parent / "lane1_data_eval"
sys.path.insert(0, str(LANE1))
from scripts.evaluation import (bootstrap_ci, paired_bootstrap_test, recall_at_k,  # noqa: E402
                                recall_at_k_hits, reciprocal_ranks, mean_reciprocal_rank)

import retrieval  # noqa: E402
from retrieval import K1, B, load_corpus, retrieve  # noqa: E402

OUT = HERE / "results"
DATASET_CSV = LANE1 / "data" / "processed" / "claimcheck_dataset.csv"
KS = (1, 3, 5, 10)


def lane1_naive_bm25(claims, corpus):
    """Lane 1's demo baseline (scripts/evaluation.py::_bm25_ranked_lists), reproduced for comparison."""
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([t.lower().split() for t in corpus["text"]])
    ids = corpus["evidence_id"].tolist()
    return [[ids[i] for i in np.argsort(-bm.get_scores(c.lower().split()))[:10]] for c in claims]


def dev_sweep(dev, corpus):
    """Check whether non-default settings would clearly beat k1=1.2, b=0.75 on the DEV split."""
    from rank_bm25 import BM25Okapi
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP
    stem = retrieval._STEM
    ids = corpus["evidence_id"].tolist()
    gold = dev["gold_evidence_id"].tolist()

    def tok(t, sw, st):
        x = re.findall(r"[a-z0-9]+", _unescape(t).lower())
        if sw:
            x = [w for w in x if w not in STOP]
        if st and stem:
            x = [stem(w) for w in x]
        return x

    rows = []
    for (sw, st), k1, b in itertools.product([(1, 1), (1, 0), (0, 1), (0, 0)], [0.5, 0.9, 1.2, 1.5, 2.0],
                                             [0.3, 0.5, 0.75, 1.0]):
        bm = BM25Okapi([tok(t, sw, st) for t in corpus["text"]], k1=k1, b=b)
        ranked = [[ids[i] for i in np.argsort(-bm.get_scores(tok(c, sw, st)), kind="stable")[:10]]
                  for c in dev["claim"]]
        rows.append(dict(stopwords=bool(sw), stemming=bool(st), k1=k1, b=b,
                         recall_at_1=recall_at_k(ranked, gold, 1), recall_at_5=recall_at_k(ranked, gold, 5),
                         mrr=mean_reciprocal_rank(ranked, gold)))
    return pd.DataFrame(rows).sort_values("mrr", ascending=False)


def _unescape(t):
    for a, b in retrieval._FEVER_ESCAPES.items():
        t = t.replace(a, b)
    return t


def score(ranked, gold):
    res, per = {}, {}
    for k in KS:
        hits = recall_at_k_hits(ranked, gold, k)
        res[f"Recall@{k}"], per[f"Recall@{k}"] = bootstrap_ci(hits), hits
    rr = reciprocal_ranks(ranked, gold)
    res["MRR"], per["MRR"] = bootstrap_ci(rr), rr
    return res, per


def fmt(ci):
    return f"{ci['point_estimate']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = list(next(iter(results.values())).keys())
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    colors = ["#d98a3d", "#2f6690", "#9aa5b1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (name, r) in enumerate(results.items()):
        v = [r[m]["point_estimate"] for m in metrics]
        err = [[r[m]["point_estimate"] - r[m]["ci_lower"] for m in metrics],
               [r[m]["ci_upper"] - r[m]["point_estimate"] for m in metrics]]
        ax.bar(x + (j - (len(results) - 1) / 2) * w, v, w, yerr=err, capsize=3, label=name, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (test split)")
    ax.set_title("Lane 2 traditional retrieval on ClaimCheck test split (95% bootstrap CI)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "lane2_retrieval_ci.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(DATASET_CSV)
    corpus = load_corpus()
    dev = data[(data["split"] == "dev") & data["gold_evidence_id"].notna()].reset_index(drop=True)
    test = data[(data["split"] == "test") & data["gold_evidence_id"].notna()].reset_index(drop=True)
    gold = test["gold_evidence_id"].tolist()

    # 1. dev sweep
    sweep = dev_sweep(dev, corpus)
    sweep.to_csv(OUT / "dev_tuning_sweep.csv", index=False)
    default = sweep[(sweep.stopwords) & (sweep.stemming) & (sweep.k1 == K1) & (sweep.b == B)].iloc[0]
    best = sweep.iloc[0]

    # 2. test evaluation
    ranked = {
        "Lane 2 BM25": [retrieve(c, 10) for c in test["claim"]],
        "Lane 2 TF-IDF": [retrieve(c, 10, method="tfidf") for c in test["claim"]],
        "Lane 1 naive BM25 (reference)": lane1_naive_bm25(test["claim"], corpus),
    }
    results, per = {}, {}
    for name, r in ranked.items():
        results[name], per[name] = score(r, gold)

    tests = {
        "Lane 2 BM25 vs Lane 1 naive BM25": ("Lane 2 BM25", "Lane 1 naive BM25 (reference)"),
        "Lane 2 BM25 vs Lane 2 TF-IDF": ("Lane 2 BM25", "Lane 2 TF-IDF"),
    }
    sig = {f"{t} ({m})": paired_bootstrap_test(per[a][m], per[b][m])
           for t, (a, b) in tests.items() for m in ("Recall@1", "Recall@5", "MRR")}

    # 3. per-claim ranks
    def rank_of(r, g):
        return r.index(g) + 1 if g in r else None
    pc = test[["claim_id", "claim", "label", "gold_evidence_id"]].copy()
    pc["bm25_rank"] = [rank_of(r, g) for r, g in zip(ranked["Lane 2 BM25"], gold)]
    pc["tfidf_rank"] = [rank_of(r, g) for r, g in zip(ranked["Lane 2 TF-IDF"], gold)]
    pc["bm25_top1"] = [r[0] for r in ranked["Lane 2 BM25"]]
    pc.to_csv(OUT / "per_claim_test.csv", index=False)

    # 4. outputs
    table = pd.DataFrame({name: {m: fmt(ci) for m, ci in r.items()} for name, r in results.items()})
    table.to_csv(OUT / "lane2_results.csv")
    plot(results)

    miss10 = pc["bm25_rank"].isna().sum()
    second = (pc["bm25_rank"] == 2).sum()
    text = corpus.set_index("evidence_id")["text"]
    words = lambda e: set(retrieval.tokenize(text[e]))  # noqa: E731
    r2 = pc[pc["bm25_rank"] == 2]
    overlap = [len(words(g) & words(t)) / max(1, len(words(g) | words(t)))
               for g, t in zip(r2["gold_evidence_id"], r2["bm25_top1"])]
    twin_share = float(np.mean([o >= 0.7 for o in overlap])) if overlap else 0.0
    md = [
        "# Lane 2 — Traditional retrieval results (scored with Lane 1's evaluation harness)", "",
        f"Corpus: {len(corpus)} evidence sentences (`evidence_corpus.csv`). "
        f"Test split: {len(test)} claims with a gold evidence id (NEI claims excluded, as the harness requires).", "",
        f"## Test split: Recall@k and MRR, 95% bootstrap CI (n={len(test)})", "",
        table.to_markdown(), "",
        "## Paired bootstrap significance tests", "",
        "| Comparison | mean A | mean B | diff | p | significant (0.05) |", "|---|---|---|---|---|---|",
    ]
    md += [f"| {k} | {v['mean_a']:.3f} | {v['mean_b']:.3f} | {v['observed_diff']:+.3f} | {('<0.001' if v['p_value'] < 0.001 else f"{v['p_value']:.3f}")} | "
           f"{'yes' if v['significant_at_0.05'] else 'no'} |" for k, v in sig.items()]
    md += [
        "", "## Parameter check on the dev split", "",
        f"{len(sweep)} settings tried (stopwords on/off x stemming on/off x k1 in {{0.5..2.0}} x b in {{0.3..1.0}}).",
        f"- Submitted default (stopwords + stemming, k1={K1}, b={B}): dev MRR {default.mrr:.3f}, "
        f"Recall@1 {default.recall_at_1:.3f}, Recall@5 {default.recall_at_5:.3f}",
        f"- Best setting (stopwords={best.stopwords}, stemming={best.stemming}, k1={best.k1}, b={best.b}): "
        f"dev MRR {best.mrr:.3f}, Recall@1 {best.recall_at_1:.3f}, Recall@5 {best.recall_at_5:.3f}",
        f"- Gap: {best.mrr - default.mrr:+.3f} MRR, so the standard defaults were kept (no test-set tuning).", "",
        "## Where BM25 goes wrong (test split)", "",
        f"- Gold evidence at rank 2: {second} / {len(test)} claims. In {twin_share:.0%} of these, the rank-1 sentence "
        "shares at least 70% of its words with the gold sentence: a near-identical contrastive twin "
        "(FEVER-Symmetric pairs often differ by one or two words, e.g. 'English-language' vs 'Spanish-language'). "
        "Bag-of-words scoring cannot tell which twin the claim agrees with.",
        f"- Gold evidence not in top-10: {miss10} / {len(test)} claims.", "",
        "Per-claim ranks: `per_claim_test.csv`. Chart: `lane2_retrieval_ci.png`.",
    ]
    (OUT / "lane2_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
