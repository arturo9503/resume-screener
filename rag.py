"""
Simple RAG (Retrieval-Augmented Generation) for resume search.

RAG in 3 steps:
  1. Index    – embed every resume into a vector once, save to disk
  2. Retrieve – embed the user's query, find the closest resume vectors
  3. Generate – pass the retrieved resumes as context to the LLM (done in app.py)
"""
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80 MB), fast, good semantic quality
CACHE_FILE = "embeddings_cache.npy"
MAX_TEXT_CHARS = 2000             # truncate each resume before embedding


class ResumeRAG:
    def __init__(self):
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Index
    # ------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame, text_col: str) -> None:
        """Encode all resumes and save the vectors to disk.

        This is the expensive step — runs once, then load_index() is used.
        """
        model = self._get_model()
        texts = self._make_texts(df, text_col)

        print(f"Encoding {len(texts)} resumes with {MODEL_NAME}...")
        self._embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # unit-normalised → cosine sim = dot product
        )
        np.save(CACHE_FILE, self._embeddings)
        self._records = df[["ID", "Category", text_col]].to_dict("records")
        print(f"Index saved to {CACHE_FILE}")

    def load_index(self, df: pd.DataFrame, text_col: str) -> bool:
        """Load a previously saved index from disk.

        Returns False when the cache is missing or doesn't match the current CSV.
        """
        path = Path(CACHE_FILE)
        if not path.exists():
            return False

        emb = np.load(path)
        if len(emb) != len(df):  # CSV was updated → rebuild
            return False

        self._get_model()         # need the model for query-time encoding
        self._embeddings = emb
        self._records = df[["ID", "Category", text_col]].to_dict("records")
        return True

    # ------------------------------------------------------------------
    # Step 2: Retrieve
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return the k resumes most semantically similar to *query*.

        Each result dict has: score, ID, Category, and the text column.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        # Embed the query with the same normalisation used for documents
        q_emb = self._get_model().encode([query], normalize_embeddings=True)

        # Cosine similarity = dot product when both sides are already normalised
        scores = (self._embeddings @ q_emb.T).squeeze()

        top_idx = np.argsort(scores)[-k:][::-1]
        return [{"score": float(scores[i]), **self._records[i]} for i in top_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the embedding model (downloaded on first use)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    @staticmethod
    def _make_texts(df: pd.DataFrame, text_col: str) -> list[str]:
        """Combine category + resume text into a single string per resume."""
        return [
            f"Category: {row['Category']}\n{str(row[text_col])[:MAX_TEXT_CHARS]}"
            for _, row in df.iterrows()
        ]
