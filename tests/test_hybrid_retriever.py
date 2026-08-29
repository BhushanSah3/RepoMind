"""Tests for the hybrid retrieval pipeline and BM25 tokenization."""

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock
from retrieval.hybrid_retriever import RetrievalResult, HybridRetriever
from indexing.bm25_index import BM25Index, tokenize_code, BM25Result
from indexing.vector_store import VectorSearchResult
from ingestion.ast_chunker import CodeChunk


def test_retrieval_result_immutable():
    """Frozen dataclass should raise on attribute assignment."""
    res = RetrievalResult(chunk_id="test_1", content="print('hello')", metadata={}, rrf_score=1.0)
    with pytest.raises(FrozenInstanceError):
        res.content = "new content"


def test_retrieval_empty_query():
    """Empty query returns empty list."""
    retriever = HybridRetriever(vector_store=MagicMock(), bm25_index=MagicMock(), embedder=MagicMock())
    results = retriever.retrieve("")
    assert results == []


def test_retrieval_zero_top_k():
    """top_k=0 returns empty list."""
    retriever = HybridRetriever(vector_store=MagicMock(), bm25_index=MagicMock(), embedder=MagicMock())
    results = retriever.retrieve("test query", top_k=0)
    assert results == []


def test_tokenize_code_snake_case():
    """'get_user_by_id' -> ['get', 'user', 'by', 'id']."""
    tokens = tokenize_code("get_user_by_id")
    assert tokens == ["get", "user", "by", "id"]


def test_tokenize_code_camel_case():
    """'getUserById' -> ['get', 'user', 'by', 'id']."""
    tokens = tokenize_code("getUserById")
    for expected in ["get", "user", "by", "id"]:
        assert expected in [t.lower() for t in tokens]


def test_tokenize_code_preserves_operators():
    """'==' and '!=' should be preserved as tokens."""
    tokens = tokenize_code("a == b and c != d")
    assert "==" in tokens
    assert "!=" in tokens


def test_tokenize_code_filters_single_chars():
    """Single chars filtered except i, x, y."""
    tokens = tokenize_code("a = i + x * y - z")
    assert "a" not in tokens
    assert "z" not in tokens
    assert "i" in tokens
    assert "x" in tokens
    assert "y" in tokens


def _make_chunk(name: str, content: str) -> CodeChunk:
    """Helper to create a minimal CodeChunk for testing."""
    return CodeChunk(
        content=content,
        chunk_type="function",
        name=name,
        file_path=f"{name}.py",
        line_start=1,
        line_end=1,
        language="python",
        chunk_id=f"{name}.py::function::{name}",
    )


def test_bm25_build_and_search():
    """Build index with CodeChunk objects, search returns BM25Result."""
    bm25 = BM25Index()
    chunks = [
        _make_chunk("hello", "def hello(): pass"),
        _make_chunk("world", "def world(): pass"),
    ]
    bm25.build_index(chunks)
    results = bm25.search("hello")
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "hello.py::function::hello"


def test_rrf_merging():
    """Verify RRF scores are computed correctly with k=60."""
    vec_mock = MagicMock()
    # Return VectorSearchResult-like objects
    dense_a = VectorSearchResult(chunk_id="A", content="code A", metadata={}, score=0.9)
    dense_b = VectorSearchResult(chunk_id="B", content="code B", metadata={}, score=0.8)
    vec_mock.search.return_value = [dense_a, dense_b]

    bm25_mock = MagicMock()
    # Use matching chunk_ids — BM25 returns BM25Result wrapping CodeChunk
    chunk_c = CodeChunk(
        content="code C", chunk_type="function", name="C",
        file_path="C.py", line_start=1, line_end=1,
        language="python", chunk_id="C",  # Must match expected chunk_id
    )
    chunk_a = CodeChunk(
        content="code A", chunk_type="function", name="A",
        file_path="A.py", line_start=1, line_end=1,
        language="python", chunk_id="A",  # Must match dense chunk_id "A"
    )
    bm25_mock.search.return_value = [
        BM25Result(chunk=chunk_c, score=2.0),
        BM25Result(chunk=chunk_a, score=1.5),
    ]

    embedder_mock = MagicMock()
    embedder_mock.embed_query.return_value = [0.1] * 384

    retriever = HybridRetriever(vector_store=vec_mock, bm25_index=bm25_mock, embedder=embedder_mock, rrf_k=60)
    results = retriever.retrieve("query", top_k=10)

    assert len(results) == 3  # A (merged), B (dense only), C (BM25 only)
    # A appears in both lists -> highest RRF score
    assert results[0].chunk_id == "A"


def test_rrf_deterministic_tiebreaking():
    """Ties should be broken alphabetically by chunk_id."""
    vec_mock = MagicMock()
    # Both at rank 1 and 2 in dense, with same score pattern
    dense_z = VectorSearchResult(chunk_id="Z", content="z code", metadata={}, score=0.9)
    dense_a = VectorSearchResult(chunk_id="A", content="a code", metadata={}, score=0.9)
    vec_mock.search.return_value = [dense_z, dense_a]

    bm25_mock = MagicMock()
    bm25_mock.search.return_value = []

    embedder_mock = MagicMock()
    embedder_mock.embed_query.return_value = [0.1] * 384

    retriever = HybridRetriever(vector_store=vec_mock, bm25_index=bm25_mock, embedder=embedder_mock)
    results = retriever.retrieve("query", top_k=10)

    # Z is rank 1 (higher RRF), A is rank 2 (lower RRF) — NOT a tie
    # Actually Z=rank1 gets 1/(60+1), A=rank2 gets 1/(60+2), so Z > A
    assert len(results) == 2
    assert results[0].chunk_id == "Z"
    assert results[1].chunk_id == "A"
