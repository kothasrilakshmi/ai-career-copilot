# app.py
from dotenv import load_dotenv
import os
import re
from pypdf import PdfReader
import streamlit as st

# --- Load API key from .env and init OpenAI client ---
load_dotenv()
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No API key found. Make sure .env contains: OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=api_key)

# ---------------- Helpers ----------------
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

def clean_text(t: str) -> str:
    """Light cleanup: collapse spaces, remove overly long runs, strip weird control chars."""
    if not t:
        return ""
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\u200b|\ufeff", "", t)
    return t.strip()

def analyze_resume_with_ai(resume_text: str, job_description: str) -> str:
    """
    Calls the OpenAI Chat Completions API and returns a structured Markdown analysis.
    """
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
st.set_page_config(page_title="Sri's AI Career Copilot", page_icon="🧭")
st.title("🧭 AI Career Copilot")
st.write("Upload your resume and job description to get personalized career recommendations.")

st.markdown("Upload your **resume (PDF)** and paste a **job description**. We'll analyze fit in the next step.")
resume_file = st.file_uploader("Resume (PDF only)", type=["pdf"])
job_description = st.text_area("Job Description", height=220, placeholder="Paste the full JD here...")

with st.expander("What did you provide? (quick preview)"):
    st.write("Resume file:", resume_file.name if resume_file else "None uploaded yet")
    st.write("Job description length:", len(job_description.strip()), "characters")

# --- Parse button: extracts & stores text into session_state ---
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

                # Persist for next steps
                st.session_state["resume_text"] = text
                st.session_state["job_description"] = job_description.strip()

            except Exception as e:
                st.error(f"Resume parsing failed: {e}")

# --- AI Analysis (always rendered; button enabled when ready) ---
st.divider()
st.subheader("AI Analysis")

ready = bool(st.session_state.get("resume_text")) and bool(st.session_state.get("job_description"))
if not ready:
    st.caption("Upload a PDF, paste the job description, and click **Continue → Parse Resume** to enable this.")

if st.button("Analyze with AI", disabled=not ready):
    with st.spinner("Analyzing resume vs job description…"):
        try:
            md = analyze_resume_with_ai(
                st.session_state["resume_text"],
                st.session_state["job_description"],
            )
            st.markdown(md)
        except Exception as e:
            st.error(f"OpenAI analysis failed: {e}")
