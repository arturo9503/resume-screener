"""
Run this once to build the search index:

    python build_index.py

Produces embeddings_cache.npy, which app.py loads at startup.
Re-run whenever the resumes table changes.
"""
from dotenv import load_dotenv
from rag import ResumeRAG
import db

load_dotenv()

df = db.load_resumes()
text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"

rag = ResumeRAG()
rag.build_index(df, text_col)

print("Done. Run `streamlit run app.py` to start the app.")
