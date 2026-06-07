"""
Run this once to build the search index:

    python build_index.py

Produces embeddings_cache.npy, which app.py loads at startup.
Re-run whenever Resume.csv changes.
"""
import os
import pandas as pd
from rag import ResumeRAG

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resume.csv")

df = pd.read_csv(CSV_PATH)
text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"

rag = ResumeRAG()
rag.build_index(df, text_col)

print("Done. Run `streamlit run app.py` to start the app.")
