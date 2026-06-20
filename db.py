import os
import pandas as pd
from sqlalchemy import create_engine

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
