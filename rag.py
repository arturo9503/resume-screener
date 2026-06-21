"""
Simple RAG (Retrieval-Augmented Generation) for resume search.

RAG in 3 steps:
  1. Index    – embed every resume into a vector once, save to PostgreSQL
  2. Retrieve – embed the user's query, find the closest vectors via pgvector
  3. Generate – pass the retrieved resumes as context to the LLM (done in app.py)
"""
import pandas as pd
import db

MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80 MB), fast, good semantic quality
MAX_TEXT_CHARS = 2000             # truncate each resume before embedding


class ResumeRAG:
    def __init__(self):
        self._model = None
        self._ready = False

    # ------------------------------------------------------------------
    # Step 1: Index
    # ------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame, text_col: str) -> None:
        """Encode all resumes and save the vectors to PostgreSQL."""
        model = self._get_model()
        texts = self._make_texts(df, text_col)

        print(f"Encoding {len(texts)} resumes with {MODEL_NAME}...")
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # unit-normalised → cosine sim = dot product
        )

        records = [
            {"id": int(row["ID"]), "embedding": embeddings[i]}
            for i, (_, row) in enumerate(df.iterrows())
        ]
        print("Saving embeddings to PostgreSQL...")
        db.save_embeddings(records)
        self._ready = True
        print("Done.")

    def load_index(self) -> bool:
        """Check if embeddings exist in PostgreSQL."""
        if not db.embeddings_exist():
            return False
        self._get_model()
        self._ready = True
        return True

    # ------------------------------------------------------------------
    # Step 2: Retrieve
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10, categories: list[str] | None = None) -> list[dict]:
        """Return the k resumes most semantically similar to *query*."""
        if not self._ready:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        q_emb = self._get_model().encode([query], normalize_embeddings=True)[0]
        results = db.search_resumes(q_emb, k, categories)
        return results.to_dict("records")

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
