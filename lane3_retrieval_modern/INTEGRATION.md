# Lane 3 → Lane 5 integration notes

## Setup (once)
```bash
pip install -r lane3_retrieval_modern/requirements.txt     # sentence-transformers, faiss-cpu, torch
cd lane3_retrieval_modern && python build_index.py          # optional: otherwise built on first call
```
First call downloads two models from Hugging Face (~420 MB + ~90 MB) and caches them.
Index is rebuilt automatically if `evidence_corpus.csv` changes, so evidence_ids always match Lane 1's corpus (same ids as Lane 2).

## Functions
| Function | Input | Output |
|---|---|---|
| `sbert_faiss.retrieve(claim, k)` | `str`, `int` | `list[str]` evidence_ids, best-first |
| `sbert_faiss.retrieve_with_scores(claim, k)` | `str`, `int` | `[(evidence_id, cosine, text), ...]` — **same tuple shape as Lane 2** |
| `reranker.rerank(claim, candidates, top_n)` | `str`, `list[str]`, `int` | `list[str]` subset of candidates, best-first (unknown/duplicate ids dropped) |
| `reranker.rerank_with_scores(claim, candidates, top_n)` | same | `[(evidence_id, rerank_score, text), ...]` |
| `reranker.search(claim, top_n=5, first_stage="sbert", pool_k=POOL_K)` | `first_stage` = `"sbert"` or `"bm25"` | `[{"evidence_id", "text", "rank", "rerank_score", "retrieval_score", "retrieval_rank"}, ...]` |

Text returned is already cleaned (`-LRB-` → `(`), same as Lane 2's `clean_text`.

## Drop-in for app.py
```python
from lane3_retrieval_modern.sbert_faiss import get_index as sbert_index, retrieve_with_scores as sbert_rws
from lane3_retrieval_modern.reranker import get_cross_encoder, search as lane3_search


@st.cache_resource(show_spinner=False)
def load_lane3():
    sbert_index()          # loads SBERT + FAISS once
    get_cross_encoder()    # loads cross-encoder once


def retrieve_evidence(claim, k=5, method="sbert_rerank"):
    if method == "bm25":
        ...                                    # existing Lane 2 code
    load_lane3()
    if method == "sbert":
        return sbert_rws(claim, k)             # [(eid, cosine, text)]
    return [(d["evidence_id"], d["rerank_score"], d["text"])   # SBERT -> rerank
            for d in lane3_search(claim, top_n=k, first_stage="sbert")]
    # BM25 -> rerank: lane3_search(claim, top_n=k, first_stage="bm25")
```

## Gotchas
- **Don't reuse the BM25 `score > 0` filter.** SBERT cosine is in [-1, 1] and rerank_score in (0, 1); neither means "relevant" vs "not". Show all top-k and let Lane 4's NEI handle weak evidence.
- Scores are on different scales per method — label them in the UI ("similarity" / "relevance"), don't compare across methods.
- `rerank_score` is relevance, not truth probability — the verdict confidence must come from Lane 4.
- Per-claim latency for each step is measured and reported in `results/lane3_results.md` ("Speed").
