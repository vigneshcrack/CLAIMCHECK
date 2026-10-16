"""
sample_claims.py — Lane 3 sanity check on the SAME 5 claims Lane 2 used
(../lane2_retrieval_traditional/results/sample_claim_ids.txt), side by side with Lane 2's BM25.

    python sample_claims.py
    python sample_claims.py --ids test_123 test_456 ...

Writes results/sample_5_claims.md and results/sample_5_claims.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reranker  # noqa: E402
from sbert_faiss import CORPUS_CSV, SBERT_MODEL, retrieve_with_scores  # noqa: E402

DATASET_CSV = CORPUS_CSV.parent / "claimcheck_dataset.csv"
LANE2_IDS = ROOT / "lane2_retrieval_traditional" / "results" / "sample_claim_ids.txt"
OUT = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    ids = a.ids or [l.strip() for l in LANE2_IDS.read_text().splitlines() if l.strip()]

    from lane2_retrieval_traditional.retrieval import retrieve as bm25_retrieve

    df = pd.read_csv(DATASET_CSV)
    sample = df[df["claim_id"].isin(ids)].set_index("claim_id").loc[ids].reset_index()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    md = ["# Lane 3 — sanity check on Lane 2's 5 sample claims (top-5)", "",
          f"SBERT `{SBERT_MODEL}` + FAISS; "
          f"rerank = cross-encoder `{reranker.CROSS_ENCODER_MODEL}` over the SBERT top-{reranker.POOL_K}.", "",
          "## Side by side: rank of the gold evidence", "",
          "| Claim ID | Claim | Label | BM25 (Lane 2) | SBERT | SBERT + rerank |", "|---|---|---|---|---|---|"]
    detail = []
    for _, c in sample.iterrows():
        gold = c["gold_evidence_id"] if isinstance(c["gold_evidence_id"], str) else None
        bm = bm25_retrieve(c["claim"], a.k)
        sb = retrieve_with_scores(c["claim"], a.k)
        rr = reranker.search(c["claim"], top_n=a.k, first_stage="sbert", pool_k=reranker.POOL_K)

        def rk(lst):
            if gold is None:
                return "no gold evidence"
            return str(lst.index(gold) + 1) if gold in lst else f"not in top-{a.k}"
        md.append(f"| {c['claim_id']} | {c['claim']} | {c['label']} | {rk(bm)} | {rk([e for e, _, _ in sb])} | "
                  f"{rk([d['evidence_id'] for d in rr])} |")

        detail += [f"## {c['claim_id']} — gold label: {c['label']}", "", f"> {c['claim']}", "",
                   f"Gold evidence: `{gold}`" if gold else "Gold evidence: none (NOT ENOUGH INFO)", "",
                   "**SBERT + FAISS** (score = cosine similarity)", "",
                   "| rank | score | evidence_id | gold? | text |", "|---|---|---|---|---|"]
        for r, (eid, s, txt) in enumerate(sb, 1):
            detail.append(f"| {r} | {s:.3f} | {eid} | {'✓' if eid == gold else ''} | {txt[:150].replace('|', '/')} |")
            rows.append(dict(claim_id=c["claim_id"], claim=c["claim"], label=c["label"], retriever="sbert", rank=r,
                             evidence_id=eid, score=round(s, 4), is_gold=eid == gold, text=txt))
        detail += ["", "**SBERT + cross-encoder rerank** (score = rerank relevance; SBERT rank in brackets)", "",
                   "| rank | score | evidence_id | gold? | text |", "|---|---|---|---|---|"]
        for d in rr:
            detail.append(f"| {d['rank']} | {d['rerank_score']:.3f} (was {d['retrieval_rank']}) | {d['evidence_id']} | "
                          f"{'✓' if d['evidence_id'] == gold else ''} | {d['text'][:150].replace('|', '/')} |")
            rows.append(dict(claim_id=c["claim_id"], claim=c["claim"], label=c["label"], retriever="sbert_rerank",
                             rank=d["rank"], evidence_id=d["evidence_id"], score=round(d["rerank_score"], 4),
                             is_gold=d["evidence_id"] == gold, text=d["text"]))
        detail.append("")

    md += [""] + detail
    pd.DataFrame(rows).to_csv(OUT / "sample_5_claims.csv", index=False)
    (OUT / "sample_5_claims.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
