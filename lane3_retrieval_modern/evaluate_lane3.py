"""
evaluate_lane3.py — scores Lane 3 (SBERT + FAISS, cross-encoder rerank) with Lane 1's evaluation harness
and compares it against Lane 2's BM25 baseline (RQ1: does semantic retrieval beat lexical retrieval?).

    python evaluate_lane3.py                    # main run
    python evaluate_lane3.py --compare-checkpoints   # + dev-only comparison of 3 SBERT checkpoints

What it does
  1. DEV split only: choose the rerank pool depth (how many first-stage candidates the cross-encoder sees)
     from {5, 10, 20, 50} by MRR                                 -> results/dev_pool_sweep.csv
     (optional) compare SBERT checkpoints on dev                 -> results/dev_checkpoint_comparison.csv
  2. TEST split (claims with a gold_evidence_id): Recall@1/3/5/10 + MRR, 95% bootstrap CI, for
       - Lane 2 BM25 (traditional baseline, imported from lane2_retrieval_traditional)
       - Lane 3 SBERT + FAISS
       - Lane 3 SBERT + FAISS -> cross-encoder rerank   (the submitted modern pipeline)
       - Lane 2 BM25 -> Lane 3 cross-encoder rerank      (isolates the reranker's contribution)
     plus paired bootstrap significance tests                    -> results/lane3_results.md / .csv
  3. Per-claim gold ranks for every system (for Lane 1's error analysis) -> results/per_claim_test.csv
  4. Retrieved evidence for ALL test claims incl. NEI, top-10, for Lane 1's ablation tiers
     (top-1 / top-5 retrieved) and Lane 4's verification runs   -> results/retrieved_test_top10.csv
  5. Bar chart with CI error bars                                -> results/lane3_vs_lane2_ci.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LANE1 = ROOT / "lane1_data_eval"
for p in (str(HERE), str(ROOT), str(LANE1)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.evaluation import (bootstrap_ci, paired_bootstrap_test,  # noqa: E402
                                recall_at_k_hits, reciprocal_ranks, mean_reciprocal_rank, recall_at_k)

import sbert_faiss  # noqa: E402
import reranker  # noqa: E402
from sbert_faiss import SBERT_MODEL, load_corpus  # noqa: E402

OUT = HERE / "results"
DATASET_CSV = LANE1 / "data" / "processed" / "claimcheck_dataset.csv"
KS = (1, 3, 5, 10)
POOLS = (5, 10, 20, 50)
CHECKPOINTS = ("sentence-transformers/all-mpnet-base-v2",
               "sentence-transformers/all-MiniLM-L6-v2",
               "sentence-transformers/multi-qa-mpnet-base-cos-v1")


def lane2_bm25(claims, k):
    from lane2_retrieval_traditional import retrieval as lane2
    return [lane2.retrieve(c, k, method="bm25") for c in claims]


def ce_scores(claims, pools):
    """Cross-encoder logit for every (claim, candidate) — computed once, reused for every pool depth."""
    return [reranker.score_pairs(c, p) for c, p in zip(claims, pools)]


def rerank_from_scores(pools, scores, depth, top_n=10):
    out = []
    for p, s in zip(pools, scores):
        p, s = p[:depth], s[:depth]
        out.append([p[i] for i in np.argsort(-s, kind="stable")[:top_n]])
    return out


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


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = list(next(iter(results.values())).keys())
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    colors = ["#9aa5b1", "#2f6690", "#3a7d44", "#d98a3d"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (name, r) in enumerate(results.items()):
        v = [r[m]["point_estimate"] for m in metrics]
        err = [[r[m]["point_estimate"] - r[m]["ci_lower"] for m in metrics],
               [r[m]["ci_upper"] - r[m]["point_estimate"] for m in metrics]]
        ax.bar(x + (j - (len(results) - 1) / 2) * w, v, w, yerr=err, capsize=3, label=name, color=colors[j % 4])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (test split)")
    ax.set_title("Lexical vs semantic retrieval on ClaimCheck test split (95% bootstrap CI)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "lane3_vs_lane2_ci.png", dpi=160)
    plt.close(fig)


def rank_of(r, g):
    return r.index(g) + 1 if isinstance(g, str) and g in r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-checkpoints", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET_CSV)
    corpus = load_corpus()
    dev = data[(data["split"] == "dev") & data["gold_evidence_id"].notna()].reset_index(drop=True)
    test_all = data[data["split"] == "test"].reset_index(drop=True)
    test = test_all[test_all["gold_evidence_id"].notna()].reset_index(drop=True)
    gold, dev_gold = test["gold_evidence_id"].tolist(), dev["gold_evidence_id"].tolist()
    maxpool = max(POOLS)

    # ---- 1. dev: checkpoint comparison (optional) and rerank pool depth ---------------------------
    if a.compare_checkpoints:
        rows = []
        for m in CHECKPOINTS:
            t0 = time.time()
            r = sbert_faiss.retrieve_batch(dev["claim"].tolist(), 10, model_name=m)
            enc = sbert_faiss.get_index(m)
            rows.append(dict(checkpoint=m, dim=enc.dim, recall_at_1=recall_at_k(r, dev_gold, 1),
                             recall_at_5=recall_at_k(r, dev_gold, 5), mrr=mean_reciprocal_rank(r, dev_gold),
                             seconds_index_plus_queries=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(OUT / "dev_checkpoint_comparison.csv", index=False)

    dev_pools = {"sbert": sbert_faiss.retrieve_batch(dev["claim"].tolist(), maxpool),
                 "bm25": lane2_bm25(dev["claim"], maxpool)}
    sweep = []
    for fs, pools in dev_pools.items():
        sc = ce_scores(dev["claim"], pools)
        for d in POOLS:
            r = rerank_from_scores(pools, sc, d)
            sweep.append(dict(first_stage=fs, pool_depth=d, recall_at_1=recall_at_k(r, dev_gold, 1),
                              recall_at_5=recall_at_k(r, dev_gold, 5), mrr=mean_reciprocal_rank(r, dev_gold),
                              pool_recall=recall_at_k(pools, dev_gold, d)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "dev_pool_sweep.csv", index=False)
    best_pool = {fs: int(g.sort_values(["mrr", "pool_depth"], ascending=[False, True]).iloc[0]["pool_depth"])
                 for fs, g in sweep.groupby("first_stage")}

    # ---- 2. test evaluation ------------------------------------------------------------------------
    claims = test_all["claim"].tolist()          # all test claims (NEI too) for the export
    timing = {}
    # BM25 baseline scored with retrieve(c, 10) exactly as Lane 2 did (its tie-breaking depends on k),
    # so the BM25 row reproduces Lane 2's published numbers; the deeper pool is only fed to the reranker.
    t0 = time.time(); bm25_top10 = lane2_bm25(claims, 10); timing["Lane 2 BM25"] = time.time() - t0
    bm25_pool = lane2_bm25(claims, maxpool)
    t0 = time.time(); sbert_pool = sbert_faiss.retrieve_batch(claims, maxpool); timing["Lane 3 SBERT"] = time.time() - t0
    t0 = time.time(); sbert_sc = ce_scores(claims, [p[:best_pool["sbert"]] for p in sbert_pool])
    timing["cross-encoder (SBERT pool)"] = time.time() - t0
    bm25_sc = ce_scores(claims, [p[:best_pool["bm25"]] for p in bm25_pool])

    ranked_all = {
        "Lane 2 BM25": bm25_top10,
        "Lane 3 SBERT": [p[:10] for p in sbert_pool],
        "Lane 3 SBERT + rerank": rerank_from_scores(sbert_pool, sbert_sc, best_pool["sbert"]),
        "Lane 2 BM25 + Lane 3 rerank": rerank_from_scores(bm25_pool, bm25_sc, best_pool["bm25"]),
    }
    has_gold = test_all["gold_evidence_id"].notna().to_numpy()
    ranked = {n: [r for r, h in zip(v, has_gold) if h] for n, v in ranked_all.items()}

    results, per = {}, {}
    for name, r in ranked.items():
        results[name], per[name] = score(r, gold)

    comparisons = {
        "RQ1: SBERT vs BM25": ("Lane 3 SBERT", "Lane 2 BM25"),
        "SBERT + rerank vs SBERT": ("Lane 3 SBERT + rerank", "Lane 3 SBERT"),
        "SBERT + rerank vs BM25": ("Lane 3 SBERT + rerank", "Lane 2 BM25"),
        "BM25 + rerank vs BM25": ("Lane 2 BM25 + Lane 3 rerank", "Lane 2 BM25"),
        "SBERT + rerank vs BM25 + rerank": ("Lane 3 SBERT + rerank", "Lane 2 BM25 + Lane 3 rerank"),
    }
    sig = {(t, m): paired_bootstrap_test(per[x][m], per[y][m])
           for t, (x, y) in comparisons.items() for m in ("Recall@1", "Recall@5", "MRR")}

    # ---- 3. per-claim ranks ------------------------------------------------------------------------
    pc = test[["claim_id", "claim", "label", "gold_evidence_id"]].copy()
    short = {"Lane 2 BM25": "bm25", "Lane 3 SBERT": "sbert", "Lane 3 SBERT + rerank": "sbert_rerank",
             "Lane 2 BM25 + Lane 3 rerank": "bm25_rerank"}
    for n, s in short.items():
        pc[f"{s}_rank"] = [rank_of(r, g) for r, g in zip(ranked[n], gold)]
        pc[f"{s}_top1"] = [r[0] if r else None for r in ranked[n]]
    pc.to_csv(OUT / "per_claim_test.csv", index=False)

    # ---- 4. export for Lane 1 ablation tiers / Lane 4 ----------------------------------------------
    text = {e: sbert_faiss.clean_text(t) for e, t in zip(corpus["evidence_id"], corpus["text"])}
    rows = []
    for i, c in test_all.iterrows():
        g = c["gold_evidence_id"] if isinstance(c["gold_evidence_id"], str) else None
        for n in ("Lane 3 SBERT", "Lane 3 SBERT + rerank"):
            for rk, e in enumerate(ranked_all[n][i], 1):
                rows.append(dict(claim_id=c["claim_id"], claim=c["claim"], label=c["label"], gold_evidence_id=g,
                                 system=short[n], rank=rk, evidence_id=e, is_gold=(e == g), text=text[e]))
    pd.DataFrame(rows).to_csv(OUT / "retrieved_test_top10.csv", index=False)

    # ---- 5. outputs --------------------------------------------------------------------------------
    table = pd.DataFrame({n: {m: fmt(ci) for m, ci in r.items()} for n, r in results.items()})
    table.to_csv(OUT / "lane3_results.csv")
    plot(results)

    # contrastive-twin analysis: claims where BM25 put the gold sentence at rank 2
    r2 = pc[pc["bm25_rank"] == 2]
    fixed = {s: int((r2[f"{s}_rank"] == 1).sum()) for s in ("sbert", "sbert_rerank", "bm25_rerank")}
    n_claims = len(test_all)
    ms = {k: 1000 * v / n_claims for k, v in timing.items()}

    md = [
        "# Lane 3 — Modern retrieval results (scored with Lane 1's evaluation harness)", "",
        f"SBERT checkpoint: `{SBERT_MODEL}` ({sbert_faiss.get_index().dim}-d, FAISS IndexFlatIP, cosine). "
        f"Cross-encoder: `{reranker.CROSS_ENCODER_MODEL}`.",
        f"Corpus: {len(corpus)} evidence sentences. Test split: {len(test)} claims with a gold evidence id "
        "(NEI claims have no gold evidence and are excluded from retrieval metrics, as the harness requires).", "",
        "## Rerank pool depth (chosen on DEV only)", "",
        sweep.round(3).to_markdown(index=False), "",
        f"Chosen pool depth by dev MRR: SBERT -> {best_pool['sbert']}, BM25 -> {best_pool['bm25']} "
        "(ties broken toward the smaller, faster pool). Rerank output is cut to top-10.", "",
        f"## Test split: Recall@k and MRR, 95% bootstrap CI (n={len(test)})", "",
        table.to_markdown(), "",
        "## Paired bootstrap significance tests (same test claims for both systems)", "",
        "| Comparison | Metric | A | B | diff (A−B) | p | significant (0.05) |", "|---|---|---|---|---|---|---|",
    ]
    md += [f"| {t} | {m} | {v['mean_a']:.3f} | {v['mean_b']:.3f} | {v['observed_diff']:+.3f} | {pfmt(v['p_value'])} | "
           f"{'yes' if v['significant_at_0.05'] else 'no'} |" for (t, m), v in sig.items()]
    md += [
        "", "## Contrastive twins (claims where BM25 ranked the gold sentence 2nd)", "",
        f"{len(r2)} test claims. Gold moved to rank 1 by: SBERT {fixed['sbert']}, SBERT + rerank {fixed['sbert_rerank']}, "
        f"BM25 + rerank {fixed['bm25_rerank']}.", "",
        "## Speed (machine that ran this; per claim, averaged over the test split; SBERT queries encoded in batches)", "",
        "\n".join(f"- {k}: {v:.1f} ms" for k, v in ms.items()), "",
        "Files: `per_claim_test.csv` (gold rank per system), `retrieved_test_top10.csv` (top-10 evidence per test claim, "
        "incl. NEI, systems `sbert` and `sbert_rerank` — use rank 1 / ranks 1–5 for the ablation tiers), "
        "`lane3_vs_lane2_ci.png`, `dev_pool_sweep.csv`" + (", `dev_checkpoint_comparison.csv`" if a.compare_checkpoints else "") + ".",
    ]
    if a.compare_checkpoints:
        md[md.index("## Rerank pool depth (chosen on DEV only)"):md.index("## Rerank pool depth (chosen on DEV only)")] = [
            "## SBERT checkpoint comparison (DEV only, no rerank)", "",
            pd.read_csv(OUT / "dev_checkpoint_comparison.csv").round(3).to_markdown(index=False), ""]
    (OUT / "lane3_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if best_pool["sbert"] != reranker.POOL_K:
        print(f"\nNOTE: dev chose pool depth {best_pool['sbert']} for SBERT; set POOL_K = {best_pool['sbert']} "
              f"in reranker.py so the app uses the same setting (currently {reranker.POOL_K}).")


if __name__ == "__main__":
    main()
