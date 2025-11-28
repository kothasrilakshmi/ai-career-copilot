from dotenv import load_dotenv
import os
import re
import json
from pypdf import PdfReader
import streamlit as st
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Make sure .env contains: OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=api_key)

def extract_pdf_text(file) -> str:
    """Extracts text from a PDF file-like object using PyPDF."""
    try:
        reader = PdfReader(file)
        chunks = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            chunks.append(txt)
        return "\n".join(chunks)
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")
    

def validate_job_description_gpt(jd_text: str) -> tuple[bool, str]:
    """
    Uses GPT to decide if jd_text is a real job description.
    Returns (is_valid, explanation).
    """

    # Optional: cheap local filter before calling GPT (saves tokens)
    if len(jd_text.split()) < 40:
        return False, "The text is very short. A job description is usually at least a few sentences."

    system_msg = (
        "You are an experienced recruiter. "
        "Given some text, decide if it is a REAL job description for a role "
        "(e.g., Data Scientist, Software Engineer, Product Manager) "
        "or if it is something else (code, personal note, essay, random text, etc.). "
        "Be strict: if it does not clearly look like a job posting, mark it as invalid."
    )

    user_msg = f"""
Text:
\"\"\"{jd_text.strip()}\"\"\"

Your task:

1. Decide if this is a valid job description.
2. Valid job description = describes a role, responsibilities, skills/experience, and/or requirements.
3. If not valid, briefly explain why.

Respond ONLY as JSON in this format:

{{
  "is_valid": true or false,
  "reason": "short explanation in one sentence"
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        is_valid = bool(data.get("is_valid", False))
        reason = data.get("reason", "")
        return is_valid, reason
    except Exception as e:
        # If parsing fails, be conservative and treat as invalid
        return False, f"Validation failed: {e}"

def clean_text(t: str) -> str:
    """Light cleanup: collapse spaces, remove overly long runs, strip weird control chars."""
    if not t:
        return ""
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\u200b|\ufeff", "", t)
    return t.strip()

def jd_too_short(jd: str, min_words: int = 40) -> bool:
    return len(jd.split()) < min_words

def analyze_resume_with_ai(resume_text: str, job_description: str) -> str:
    """
    Calls the OpenAI Chat Completions API and returns a structured Markdown analysis.
    """

    jd_words = len(job_description.split())

    if jd_words < 40:
        # Fallback: resume-only feedback, no pretending there is a JD
        system = (
            "You are a precise career advisor. The user did NOT provide a real job "
            "description (it is missing or too short). "
            "Give feedback on the resume ALONE and explicitly say that you cannot "
            "compare against a specific JD."
        )
        user = f"""
Return your answer in **Markdown** with these sections:

1) **Overall Resume Strengths** — 3–6 bullets
2) **Areas to Improve** — 3–6 bullets
3) **Resume Bullet Rewrites (ATS-ready)** — 3–6 bullets
4) **General Professional Summary (3–4 sentences)**

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION (too short / invalid) ---
{job_description}
""".strip()
    else:
        # Normal comparison mode
        system = (
            "You are a precise career advisor. Analyze a candidate's resume against a job description. "
            "Be specific, concise, and actionable. Use clear section headers and bullet points. "
            "Do not invent facts; use only provided text."
        )
        user = f"""
Return your answer in **Markdown** with these sections:

1) **Strengths vs JD** — 3–6 bullets
2) **Skill/Experience Gaps** — 3–6 bullets (use verb–noun phrasing, e.g., "Hands-on Databricks pipelines")
3) **Resume Bullet Rewrites (ATS-ready)** — 3–6 bullets; use strong verbs + quantification placeholders if needed
4) **Tailored Professional Summary (3–4 sentences)** — role-aligned, no fluff
5) **Top Keywords to Add** — comma-separated

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------- UI ----------------

st.set_page_config(
    page_title="Sri's AI Career Copilot",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
<style>

/* Light pastel background */
.stApp {
  background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%);
  color: #1F2937;
}

h1, h2, h3, h4, h5, h6, p, label {
  color: #1F2937;
}

/* Main container centered */
.block-container {
  max-width: 1100px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}

/* Soft pastel hero banner */
.hero {
  padding: 1.8rem;
  border-radius: 1.3rem;
  background: linear-gradient(120deg, #BEE3F8, #C6F6D5, #FED7E2);
  color: #1E293B;  /* dark slate text */
  box-shadow: 0 10px 25px rgba(0,0,0,0.08);
  animation: fadeIn 0.6s ease-out;
}
.hero-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 0.4rem;
}
.hero-sub {
  opacity: 0.9;
  font-size: 1rem;
}
.pill {
  display:inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  border: 1px solid #CBD5E1;
  background: #F8FAFC;
  color: #334155;
  font-size: 0.75rem;
  margin-right: 0.4rem;
  margin-top: 0.4rem;
}

/* Pastel cards */
.card {
  border-radius: 1rem;
  padding: 1.2rem;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
.card-title {
  font-size: 1rem;
  font-weight: 600;
  color: #334155;
  margin-bottom: 0.5rem;
}

/* Soft pastel buttons */
.stButton > button {
  border-radius: 999px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  border: none;
  background: linear-gradient(135deg, #A5F3FC, #C7D2FE);
  color: #1E293B;
  box-shadow: 0 5px 15px rgba(0,0,0,0.08);
  transition: 0.15s ease-in-out;
}
.stButton > button:hover:not(:disabled) {
  filter: brightness(1.05);
  transform: translateY(-2px);
}
.stButton > button:disabled {
  background: #E2E8F0;
  color: #94A3B8;
  box-shadow: none;
}

/* Subtle caption */
.caption {
  font-size: 0.85rem;
  color: #64748B;
}

/* Animation */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

</style>

""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">🧭 AI Career Copilot</div>
  <div class="hero-sub">
    Upload your resume and a job description to get AI-powered fit analysis, ATS-friendly bullet rewrites,
    and a tailored professional summary.
  </div>
  <div>
    <span class="pill">Resume vs JD</span>
    <span class="pill">ATS focused</span>
    <span class="pill">Powered by GPT-4o mini</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")  

left_col, right_col = st.columns([0.65, 0.35])

with left_col:
    st.markdown("#### 📥 Inputs")
    st.markdown(
        '<span class="caption">Step 1 – Upload your resume PDF. Step 2 – Paste the full job description.</span>',
        unsafe_allow_html=True,
    )
    resume_file = st.file_uploader("Resume (PDF only)", type=["pdf"])
    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here...",
    )

with right_col:
    st.markdown(
        '<div class="card">'
        '<div class="card-title">Status & quick info</div>',
        unsafe_allow_html=True,
    )
    st.write("• Resume file:", resume_file.name if resume_file else "None uploaded yet")
    st.write("• JD length:", len(job_description.strip()), "characters")
    if "resume_text" in st.session_state and st.session_state["resume_text"]:
        st.success("Parsed resume stored and ready for analysis.")
    else:
        st.info("Parse your resume to continue to AI analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🧾 Step 2 – Parse your resume")

if st.button("Continue → Parse Resume"):
    if not resume_file:
        st.error("Please upload a PDF resume first.")
    elif not job_description.strip():
        st.error("Please paste the job description.")
    else:
        with st.spinner("Reading your resume PDF…"):
            try:
                resume_file.seek(0)
                raw = extract_pdf_text(resume_file)
                text = clean_text(raw)

                if not text or len(text) < 200:
                    st.warning(
                        "I could not extract much text from this PDF. "
                        "If it’s a scanned image, consider exporting a text-based PDF or sharing a DOCX."
                    )
                else:
                    st.success("Resume parsed successfully ✅")

                
                with st.expander("Preview parsed resume text"):
                    st.write(text[:2000] + ("…" if len(text) > 2000 else ""))

                
                st.session_state["resume_text"] = text
                st.session_state["job_description"] = job_description.strip()

                is_valid_jd, jd_reason = validate_job_description_gpt(
                    st.session_state["job_description"]
                )
                st.session_state["job_description_valid"] = is_valid_jd

                if is_valid_jd:
                    st.success("✅ Job description looks valid for analysis.")
                else:
                    st.error(
                        "❌ This doesn’t look like a real job description.\n\n"
                        f"Reason: {jd_reason}"
                    )

            except Exception as e:
                st.error(f"Resume parsing failed: {e}")


st.markdown("---")
st.markdown("### 🧠 Step 3 – Analyze with AI")

ready = (
    bool(st.session_state.get("resume_text")) and
    bool(st.session_state.get("job_description")) and
    bool(st.session_state.get("job_description_valid"))
)
if not ready:
    st.caption(
        "Upload a PDF, paste the job description, and click **Continue → Parse Resume** before analyzing."
    )

if st.button("Analyze with AI", disabled=not ready):
    jd = st.session_state["job_description"]
    if jd_too_short(jd):
        st.error(
            "The job description is too short or not a real JD. "
            "Please paste the full posting so I can compare your resume properly."
        )
    else:
        with st.spinner("Analyzing resume vs job description…"):
            try:
                md = analyze_resume_with_ai(
                    st.session_state["resume_text"],
                    jd,
                )
                st.markdown(md)
            except Exception as e:
                st.error(f"OpenAI analysis failed: {e}")
