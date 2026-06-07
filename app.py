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

api_key = os.getenv("ANTHROPIC_API_KEY")
demo_mode = not api_key

if demo_mode:
    st.warning("No ANTHROPIC_API_KEY found — running in demo mode. Answers are simulated.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    sample = df.sample(min(30, len(df))).reset_index(drop=True)
    resume_context = "\n\n---\n\n".join(
        f"ID: {row['ID']}, Category: {row['Category']}\n{str(row[text_col])[:1500]}"
        for _, row in sample.iterrows()
    )
    stats = (
        f"Total resumes: {len(df)}\n"
        f"Categories: {', '.join(sorted(df['Category'].unique()))}"
    )
    st.session_state.system_prompt = (
        f"You are analyzing a resume database.\n\nDatabase stats:\n{stats}\n\n"
        f"Sample resumes:\n\n{resume_context}\n\n"
        "Answer clearly and specifically based on the data above."
    )

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        del st.session_state.system_prompt
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the resumes"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if demo_mode:
        sample = df.sample(min(30, len(df)))
        cats = ", ".join(sample["Category"].value_counts().head(5).index.tolist())
        response = (
            f"**Demo answer:** Your question was *\"{prompt}\"*. "
            f"The sample of 30 resumes includes categories like {cats}. "
            "Add an API key to get real answers from Claude."
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        client = anthropic.Anthropic(api_key=api_key)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                message = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=1000,
                    system=st.session_state.system_prompt,
                    messages=st.session_state.messages,
                )
            response = message.content[0].text
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
