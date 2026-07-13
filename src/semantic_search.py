"""
semantic_search.py
------------------
BONUS: Semantic search over extracted clauses using sentence-transformers
embeddings (runs fully locally, no API key needed).

Usage (standalone):
    python -m src.semantic_search --query "termination without cause" \
        --results_json output/results.json --top_k 5
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load heavy dependencies
# ---------------------------------------------------------------------------

def _load_deps():
    """Import sentence-transformers and numpy lazily."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        return SentenceTransformer, np
    except ImportError:
        raise ImportError(
            "Semantic search requires sentence-transformers and numpy.\n"
            "Install with: pip install sentence-transformers numpy"
        )


# ---------------------------------------------------------------------------
# Embedding index builder
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, fast, great quality

_model_cache = None
_index_cache: Optional[Dict] = None


def _get_embed_model():
    global _model_cache
    if _model_cache is None:
        SentenceTransformer, _ = _load_deps()
        logger.info(f"Loading embedding model '{EMBED_MODEL_NAME}'…")
        _model_cache = SentenceTransformer(EMBED_MODEL_NAME)
    return _model_cache


def build_index(results: List[Dict[str, Any]]) -> Dict:
    """
    Build an in-memory vector index over all clause fields and summaries.

    Each 'document' in the index is one clause/summary from one contract.
    Returns a dict with:
        texts      : List[str]  – clause text
        meta       : List[dict] – contract_id, title, field
        embeddings : np.ndarray – shape (N, dim)
    """
    SentenceTransformer, np = _load_deps()
    model = _get_embed_model()

    texts, meta = [], []
    clause_fields = [
        "summary", "termination_clause",
        "confidentiality_clause", "liability_clause",
    ]

    for r in results:
        for field in clause_fields:
            val = r.get(field, "")
            if val and "Extraction failed" not in val and "Not specified" not in val:
                texts.append(val)
                meta.append({
                    "contract_id": r["contract_id"],
                    "title":       r.get("title", ""),
                    "field":       field,
                })

    if not texts:
        return {"texts": [], "meta": [], "embeddings": None}

    logger.info(f"Encoding {len(texts)} clause segments…")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return {"texts": texts, "meta": meta, "embeddings": embeddings}


def load_index_from_file(results_json: Path) -> Dict:
    """Load results JSON and build the index."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    _index_cache = build_index(results)
    return _index_cache


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    index: Dict,
    top_k: int = 5,
) -> List[Tuple[float, Dict, str]]:
    """
    Cosine similarity search over the index.

    Returns
    -------
    List of (score, meta_dict, text) tuples, descending by score.
    """
    _, np = _load_deps()
    model = _get_embed_model()

    if index["embeddings"] is None or len(index["texts"]) == 0:
        return []

    q_emb = model.encode([query], convert_to_numpy=True)   # (1, dim)
    corpus_emb = index["embeddings"]                        # (N, dim)

    # Cosine similarity = dot product of L2-normalised vectors
    q_norm      = q_emb      / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    corpus_norm = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)
    scores = (corpus_norm @ q_norm.T).squeeze()             # (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (float(scores[i]), index["meta"][i], index["texts"][i])
        for i in top_indices
    ]


def print_search_results(query: str, hits: List[Tuple]) -> None:
    """Pretty-print search results."""
    print(f"\n{'='*70}")
    print(f"Query: {query!r}")
    print(f"{'='*70}")
    if not hits:
        print("No results found.")
        return
    for rank, (score, meta, text) in enumerate(hits, 1):
        print(f"\n#{rank}  Score: {score:.4f}")
        print(f"   Contract : {meta['contract_id']} – {meta['title'][:60]}")
        print(f"   Field    : {meta['field']}")
        print(f"   Excerpt  : {text[:300]}{'…' if len(text)>300 else ''}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Semantic search over CUAD clauses")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--results_json",
        default="output/results.json",
        help="Path to results.json produced by the pipeline",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    index = load_index_from_file(Path(args.results_json))
    hits  = search(args.query, index, top_k=args.top_k)
    print_search_results(args.query, hits)
