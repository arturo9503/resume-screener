import streamlit as st
import pandas as pd
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume Q&A", layout="centered")
st.title("Resume Q&A")


@st.cache_data
def load_resumes():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resume.csv")
    return pd.read_csv(csv_path)


df = load_resumes()
text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"

st.caption(f"{len(df)} resumes loaded across {df['Category'].nunique()} categories")

question = st.text_input("Ask a question about the resumes", placeholder="e.g. How many IT resumes are there? Find candidates with Python experience.")

api_key = os.getenv("ANTHROPIC_API_KEY")
demo_mode = not api_key

if demo_mode:
    st.warning("No ANTHROPIC_API_KEY found — running in demo mode. Answers are simulated.")

if st.button("Ask", disabled=not question):
    sample = df.sample(min(30, len(df))).reset_index(drop=True)

    if demo_mode:
        cats = ", ".join(sample["Category"].value_counts().head(5).index.tolist())
        st.markdown(
            f"**Demo answer:** Your question was *\"{question}\"*. "
            f"The sample of 30 resumes includes categories like {cats}. "
            "Add an API key to get real answers from Claude."
        )
    else:
        resume_context = "\n\n---\n\n".join(
            f"ID: {row['ID']}, Category: {row['Category']}\n{str(row[text_col])[:1500]}"
            for _, row in sample.iterrows()
        )

        stats = (
            f"Total resumes: {len(df)}\n"
            f"Categories: {', '.join(sorted(df['Category'].unique()))}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        with st.spinner("Thinking..."):
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": (
                        f"You are analyzing a resume database.\n\nDatabase stats:\n{stats}\n\n"
                        f"Sample resumes:\n\n{resume_context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer clearly and specifically based on the data above."
                    )
                }]
            )
        st.markdown(message.content[0].text)
