from __future__ import annotations

from rag.retrieval.candidate import Candidate
from rag.retrieval.filters import RetrievalFilters, build_qdrant_filter
from rag.retrieval.hybrid_search import reciprocal_rank_fusion


def test_empty_filters_produce_no_qdrant_filter():
    assert build_qdrant_filter(RetrievalFilters()) is None
    assert build_qdrant_filter(None) is None


def test_filters_build_expected_conditions():
    filters = RetrievalFilters(source_type="pdf", category="policies")
    q_filter = build_qdrant_filter(filters)
    assert q_filter is not None
    keys = {c.key for c in q_filter.must}
    assert keys == {"source_type", "category"}


def _candidate(chunk_id: str, score: float, method: str = "dense") -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        source_type="web",
        retrieval_method=method,
    )


def test_rrf_favors_items_ranked_high_in_both_lists():
    dense = [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7)]
    sparse = [_candidate("a", 5.0), _candidate("c", 4.0), _candidate("b", 3.0)]

    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.5)

    assert fused[0].chunk_id == "a"  # ranked #1 in both lists
    assert {c.chunk_id for c in fused} == {"a", "b", "c"}


def test_rrf_includes_items_only_in_one_list():
    dense = [_candidate("a", 0.9)]
    sparse = [_candidate("b", 5.0)]

    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.5)
    assert {c.chunk_id for c in fused} == {"a", "b"}


def test_rrf_weight_shifts_ranking_toward_dominant_list():
    dense = [_candidate("a", 0.9), _candidate("b", 0.8)]
    sparse = [_candidate("b", 5.0), _candidate("a", 4.0)]

    fused_dense_heavy = reciprocal_rank_fusion(dense, sparse, dense_weight=0.95)
    assert fused_dense_heavy[0].chunk_id == "a"

    fused_sparse_heavy = reciprocal_rank_fusion(dense, sparse, dense_weight=0.05)
    assert fused_sparse_heavy[0].chunk_id == "b"


def test_candidate_to_source_dict_prefers_title_and_includes_url_page():
    candidate = Candidate(
        chunk_id="c1",
        document_id="d1",
        text="t",
        score=0.5,
        source_type="pdf",
        title="Refund Policy",
        url=None,
        page=3,
    )
    source = candidate.to_source_dict()
    assert source["title"] == "Refund Policy"
    assert source["page"] == 3
    assert "url" not in source
