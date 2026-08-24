"""Dense and sparse retrieval merged with reciprocal-rank fusion."""
from dataclasses import dataclass
from indexing.bm25_index import BM25Index
from indexing.embedder import LocalEmbedder
from indexing.vector_store import ChromaStore

@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    content: str
    metadata: dict
    rrf_score: float

class HybridRetriever:
    def __init__(self, vector_store: ChromaStore, bm25_index: BM25Index, embedder: LocalEmbedder, rrf_k: int = 60):
        self.vector_store, self.bm25_index, self.embedder, self.rrf_k = vector_store, bm25_index, embedder, rrf_k
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0:
            return []
        candidate_count = top_k * 2
        dense = self.vector_store.search(self.embedder.embed_query(query), top_k=candidate_count)
        sparse = self.bm25_index.search(query, top_k=candidate_count)
        scores, records = {}, {}
        for rank, item in enumerate(dense, 1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (self.rrf_k + rank)
            records[item.chunk_id] = (item.content, item.metadata)
        for rank, item in enumerate(sparse, 1):
            scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0) + 1 / (self.rrf_k + rank)
            records[item.chunk.chunk_id] = (item.chunk.content, item.chunk.model_dump(exclude={'content'}))
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return [RetrievalResult(key, records[key][0], records[key][1], score) for key, score in ranked]
