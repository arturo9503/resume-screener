import streamlit as st
import os
from dotenv import load_dotenv
import anthropic
import rag
import db
import analysis

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
    if not db.embeddings_exist():
        st.error("Search index not found. Run `python build_index.py` first.")
        st.stop()
    return rag.load_model()


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

model = init_rag()


@st.dialog("Resume", width="large")
def show_resume(r):
    st.html(r.get("Resume_html", str(r.get("Resume_str", ""))))


def get_analyses(ranked, job_description):
    if demo_mode:
        return {}
    with st.spinner("Analyzing candidates with Claude..."):
        try:
            results = analysis.analyze(ranked, job_description, api_key)
            return {str(r["candidate_id"]): r for r in results}
        except Exception as e:
            st.warning(f"Claude analysis unavailable: {e}")
            return {}


def render_candidate(i, r, a):
    cols = st.columns([4, 1, 1])
    if cols[0].button(f"#{i + 1} · {r['Category']} · ID {r['ID']}", key=f"resume_{i}"):
        show_resume(r)
    if a:
        cols[1].metric("Fit", f"{a['fit_score']}/10")
    cols[2].metric("Similarity", f"{r['score']:.3f}")
    if a:
        st.write(a["explanation"])
        st.markdown(f"**Strengths:** {a['strengths'][0]} · {a['strengths'][1]}")
        concern = a.get("concern")
        if concern and str(concern).lower() != "null":
            st.markdown(f"**Concern:** {concern}")
    st.divider()


def get_chat_response(prompt, system_prompt, results):
    if demo_mode:
        cats = ", ".join(r["Category"] for r in results[:5])
        return (
            f"**Demo answer:** Your question was *\"{prompt}\"*. "
            f"RAG retrieved {len(results)} relevant resumes including categories: {cats}. "
            "Add an API key to get real answers from Claude."
        )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=st.session_state.messages,
    )
    return message.content[0].text


if selected_posting is not None:
    st.subheader(f"{selected_posting['title']} @ {selected_posting['company_name']}")
    with st.expander("Job description"):
        st.write(selected_posting["description"])
    ranked = rag.search(model, str(selected_posting["description"]), k=25)
    analyses = get_analyses(ranked, str(selected_posting["description"]))
    if analyses:
        ranked = sorted(ranked, key=lambda r: analyses.get(str(r["ID"]), {}).get("fit_score", 0), reverse=True)
    label = "fit score" if analyses else "semantic similarity"
    st.markdown(f"**Top {len(ranked)} candidates by {label}**")
    for i, r in enumerate(ranked):
        render_candidate(i, r, analyses.get(str(r["ID"])))

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

    results = rag.search(model, prompt, k=10)
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

    with st.spinner("Thinking..."):
        response = get_chat_response(prompt, system_prompt, results)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.expander(f"Retrieved {len(results)} relevant resumes (RAG)"):
        for r in results:
            st.markdown(f"**{r['Category']}** — ID: {r['ID']} — similarity: `{r['score']:.3f}`")
