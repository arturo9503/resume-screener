import json
import anthropic


def analyze(candidates: list[dict], job_description: str, api_key: str) -> list[dict]:
    """Evaluate all candidates against a job description in a single Claude call."""
    candidate_blocks = "\n\n".join(
        f"Candidate ID: {c['ID']}\n{str(c.get('Resume_str', ''))[:2000]}"
        for c in candidates
    )

    prompt = (
        "You are evaluating candidates for a specific job role.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidates:\n{candidate_blocks}\n\n"
        "Return a JSON array with exactly one object per candidate:\n"
        "[\n"
        "  {\n"
        '    "candidate_id": "<ID from above>",\n'
        '    "fit_score": <integer 1-10>,\n'
        '    "explanation": "<one paragraph: why they are or are not a strong match for this role>",\n'
        '    "strengths": ["<strongest signal 1>", "<strongest signal 2>"],\n'
        '    "concern": "<one concern or gap, or null if none>"\n'
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- Base your analysis only on information present in the resume. Do not invent credentials or experience.\n"
        "- fit_score must be an integer from 1 to 10.\n"
        "- strengths must be exactly 2 items.\n"
        "- concern is null if you identify no significant gaps.\n"
        "- Return only the JSON array, no other text."
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    start = raw.index("[")
    end = raw.rindex("]") + 1
    return json.loads(raw[start:end])
