import os
import pandas as pd
from sqlalchemy import create_engine, text

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"])
    return _engine


def load_resumes() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(
            'SELECT "ID", "Resume_str", "Resume_html", "Category" FROM resume ORDER BY "ID"',
            conn,
        )


def load_postings() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(
            """
            SELECT job_id, company_name, title, description, location
            FROM postings
            WHERE description IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 5
            """,
            conn,
        )


def embeddings_exist() -> bool:
    with get_engine().connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM resume WHERE embedding IS NOT NULL'))
        return result.scalar() > 0


def save_embeddings(records: list[dict]) -> None:
    with get_engine().connect() as conn:
        for record in records:
            vec_str = "[" + ",".join(map(str, record["embedding"].tolist())) + "]"
            conn.execute(
                text('UPDATE resume SET embedding = CAST(:emb AS vector) WHERE "ID" = :id'),
                {"emb": vec_str, "id": record["id"]}
            )
        conn.commit()


def search_resumes(query_embedding, k: int, categories: list[str] | None = None) -> pd.DataFrame:
    vec_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
    category_filter = 'AND "Category" = ANY(:cats)' if categories else ""
    params = {"emb": vec_str, "k": k}
    if categories:
        params["cats"] = categories

    with get_engine().connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT "ID", "Category", "Resume_str", "Resume_html",
                       1 - (embedding <=> CAST(:emb AS vector)) AS score
                FROM resume
                WHERE embedding IS NOT NULL {category_filter}
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :k
            """),
            params
        )
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
