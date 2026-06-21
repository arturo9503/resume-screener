import streamlit as st
import pandas as pd
import anthropic
import os
from dotenv import load_dotenv
from rag import ResumeRAG
import db

load_dotenv()

st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("Resume Screener")


@st.cache_data
def load_resumes():
    return db.load_resumes()


@st.cache_data
def load_postings():
    return db.load_postings()


@st.cache_resource(show_spinner="Loading search index...")
def init_rag():
    """Load the pre-built RAG index. Run build_index.py first if missing."""
    rag = ResumeRAG()
    if not rag.load_index():
        st.error("Search index not found. Run `python build_index.py` first.")
        st.stop()
    return rag


df = load_resumes()
text_col = "Resume_str" if "Resume_str" in df.columns else "Resume_s"
st.caption(f"{len(df)} resumes loaded across {df['Category'].nunique()} categories")

postings_df = load_postings()

st.sidebar.header("Job Posting")
posting_labels = ["(None — use chat only)"] + [
    f"{row['title']} @ {row['company_name']}" for _, row in postings_df.iterrows()
]
selected_label = st.sidebar.selectbox("Select a posting to screen against", posting_labels)
selected_posting = (
    None
    if selected_label == "(None — use chat only)"
    else postings_df.iloc[posting_labels.index(selected_label) - 1]
)

api_key = os.getenv("ANTHROPIC_API_KEY")
demo_mode = not api_key

if demo_mode:
    st.warning("No ANTHROPIC_API_KEY found — running in demo mode. Answers are simulated.")

rag = init_rag()


@st.dialog("Resume", width="large")
def show_resume(r):
    st.html(r.get("Resume_html", str(r.get("Resume_str", ""))))


if selected_posting is not None:
    st.subheader(f"{selected_posting['title']} @ {selected_posting['company_name']}")
    with st.expander("Job description"):
        st.write(selected_posting["description"])
    ranked = rag.search(str(selected_posting["description"]), k=50)
    st.markdown(f"**Top {len(ranked)} candidates by semantic match**")
    for i, r in enumerate(ranked):
        c1, c2, c3, c4 = st.columns([1, 3, 2, 1])
        c1.write(i + 1)
        c2.write(r["Category"])
        if c3.button(f"ID {r['ID']}", key=f"resume_{i}"):
            show_resume(r)
        c4.write(f"{r['score']:.3f}")
    st.divider()

DB_STATS = (
    f"Total resumes: {len(df)}\n"
    f"Categories: {', '.join(sorted(df['Category'].unique()))}"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the resumes"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: retrieve the most relevant resumes for this specific question
    results = rag.search(prompt, k=10)

    # Build system prompt from retrieved resumes (not a random sample)
    resume_context = "\n\n---\n\n".join(
        f"ID: {r['ID']}, Category: {r['Category']}\n{str(r[text_col])[:1500]}"
        for r in results
    )
    system_prompt = (
        f"You are analyzing a resume database.\n\nDatabase stats:\n{DB_STATS}\n\n"
        f"These {len(results)} resumes were retrieved as most relevant to the current question:\n\n"
        f"{resume_context}\n\n"
        "Answer clearly and specifically based on the retrieved resumes above."
    )

    if demo_mode:
        cats = ", ".join(r["Category"] for r in results[:5])
        response = (
            f"**Demo answer:** Your question was *\"{prompt}\"*. "
            f"RAG retrieved {len(results)} relevant resumes including categories: {cats}. "
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
                    system=system_prompt,
                    messages=st.session_state.messages,
                )
            response = message.content[0].text
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Show which resumes were retrieved so the user can understand RAG
    with st.expander(f"Retrieved {len(results)} relevant resumes (RAG)"):
        for r in results:
            st.markdown(f"**{r['Category']}** — ID: {r['ID']} — similarity: `{r['score']:.3f}`")
