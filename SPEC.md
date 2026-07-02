# Resume Screener — Specification

## System Overview

The Resume Screener helps recruiters evaluate candidates against a job posting. A recruiter selects a posting; the system finds the most relevant resumes from a pre-indexed corpus; an AI model scores each candidate and explains its reasoning relative to that specific role; the UI presents candidates ranked by fit. A secondary chat interface for open-ended questions about the corpus is unaffected by these changes.

---

## Behavior

### Search
Semantic search returns the top 25 candidates by similarity — enough to give Claude meaningful signal without degrading score consistency. These candidates are the only ones passed downstream.

### AI Analysis
All top candidates and the job description are evaluated together in a single AI call. For each candidate the AI returns:
- A numeric fit score
- A plain-language explanation of why they are or aren't a strong match for this specific role
- The two strongest signals from their resume
- One concern or gap, if any

Analysis is grounded only in what is present in the resume. The AI must not invent credentials or experience.

### Display
Candidates are shown ranked by fit score. Both the AI fit score and the original similarity score remain visible. Name, explanation, strengths, and concern (if any) are displayed per candidate.

### Graceful Degradation
If the AI service is unavailable or unconfigured, the app shows a warning and falls back to similarity-only ranking. It does not error out.

---

## Constraints

- All candidates are evaluated in a single AI call, not one call per candidate.
- The fit score is a whole number on a fixed scale.
- Strengths are always exactly two items — not one, not three.
- Analysis never replaces the similarity score; both are always shown.
- The AI may return a null concern if none is identified.

---

## Out of Scope

- Streaming the AI response
- Authentication or access control
- Changing the embedding model or index pipeline
- Pagination
- Persisting analysis results
- Manual job description entry (only posted jobs from the existing dataset)
