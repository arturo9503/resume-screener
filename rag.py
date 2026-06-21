"""
Simple RAG (Retrieval-Augmented Generation) for resume search.

RAG in 3 steps:
  1. Index    – embed every resume into a vector once, save to PostgreSQL
  2. Retrieve – embed the user's query, find the closest vectors via pgvector
  3. Generate – pass the retrieved resumes as context to the LLM (done in app.py)
"""
import pandas as pd
import db

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TEXT_CHARS = 2000


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def build_index(model, df: pd.DataFrame, text_col: str) -> None:
    """Encode all resumes and save the vectors to PostgreSQL."""
    texts = [
        f"Category: {row['Category']}\n{str(row[text_col])[:MAX_TEXT_CHARS]}"
        for _, row in df.iterrows()
    ]
    print(f"Encoding {len(texts)} resumes with {MODEL_NAME}...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    records = [
        {"id": int(row["ID"]), "embedding": embeddings[i]}
        for i, (_, row) in enumerate(df.iterrows())
    ]
    print("Saving embeddings to PostgreSQL...")
    db.save_embeddings(records)
    print("Done.")


def search(model, query: str, k: int = 10, categories: list[str] | None = None) -> list[dict]:
    """Return the k resumes most semantically similar to query."""
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    return db.search_resumes(q_emb, k, categories).to_dict("records")
