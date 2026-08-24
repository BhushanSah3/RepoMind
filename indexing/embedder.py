"""Lazy local embeddings using the MiniLM sentence-transformer model."""

from typing import Any

from exceptions import RetrievalError


class LocalEmbedder:
    """CPU-friendly embeddings loaded only when the index is first used."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in bounded batches with cosine-compatible normalization."""

        if not texts:
            return []
        embeddings = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query with the same normalization as indexed text."""

        return self.embed_texts([query])[0]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RetrievalError(
                    "Embedding support is not installed. Install sentence-transformers."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        return self._model
