"""
Run this once to build the search index:

    python build_index.py

Encodes all resumes and saves embeddings to PostgreSQL.
Re-run whenever the resumes table changes.
"""
from dotenv import load_dotenv
import db
import rag

load_dotenv()

df = db.load_resumes()
text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"
model = rag.load_model()
rag.build_index(model, df, text_col)

print("Done. Run `streamlit run app.py` to start the app.")
