"""Lazy local cross-encoder reranking."""
from retrieval.hybrid_retriever import RetrievalResult

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name, self._model = model_name, None
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 8) -> list[RetrievalResult]:
        if not results or top_k <= 0: return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._model.max_length = 512
        scores = self._model.predict([(query, result.content) for result in results])
        return [item for _, item in sorted(zip(scores, results), key=lambda pair: pair[0], reverse=True)[:top_k]]
