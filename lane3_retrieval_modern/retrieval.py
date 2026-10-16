"""
retrieval.py — alias so Lane 3 also matches the file name in the lane brief.
The implementation lives in sbert_faiss.py; import from either.
"""
try:
    from .sbert_faiss import (get_index, get_text, retrieve, retrieve_batch,  # noqa: F401
                              retrieve_with_scores)
except ImportError:
    from sbert_faiss import (get_index, get_text, retrieve, retrieve_batch,  # noqa: F401
                             retrieve_with_scores)
