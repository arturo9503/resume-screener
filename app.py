import streamlit as st
import pandas as pd
import anthropic
import json
import os
import random
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")

demo_mode = st.sidebar.checkbox("Demo mode (no API key needed)", value=not os.getenv("ANTHROPIC_API_KEY"))

job_description = st.text_area(
    "Job Description",
    height=150,
    placeholder="Paste the job description here..."
)

uploaded_file = st.file_uploader("Upload Kaggle Resume CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # handle both full and truncated column names from Kaggle dataset
    text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"

    col1, col2 = st.columns(2)
    with col1:
        categories = ["All"] + sorted(df["Category"].unique().tolist())
        selected = st.selectbox("Filter by category", categories)
    with col2:
        n = st.slider("Resumes to screen", 5, 30, 10)

    if selected != "All":
        df = df[df["Category"] == selected]

    df_sample = df.sample(min(n, len(df))).reset_index(drop=True)
    st.caption(f"{len(df_sample)} resumes selected")

    if st.button("Screen Resumes", disabled=not job_description):
        client = None if demo_mode else anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        results = []
        bar = st.progress(0, text="Screening...")

        for i, row in df_sample.iterrows():
            if demo_mode:
                data = {"score": random.randint(1, 10), "reason": "Demo mode — no API call made"}
            else:
                resume_text = str(row[text_col])[:3000]
                message = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Score this resume for the role below. "
                            'Respond with JSON only: {"score": <1-10>, "reason": "<one sentence>"}\n\n'
                            f"JOB:\n{job_description}\n\nRESUME:\n{str(row[text_col])[:3000]}"
                        )
                    }]
                )
                try:
                    data = json.loads(message.content[0].text)
                except json.JSONDecodeError:
                    data = {"score": 0, "reason": "Could not parse response"}

            results.append({
                "Score": data.get("score", 0),
                "Reason": data.get("reason", ""),
                "Category": row["Category"],
                "ID": row["ID"],
            })

            bar.progress((i + 1) / len(df_sample), text=f"Screened {i + 1}/{len(df_sample)}")

        bar.empty()
        results_df = (
            pd.DataFrame(results)
            .sort_values("Score", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(results_df, use_container_width=True)
