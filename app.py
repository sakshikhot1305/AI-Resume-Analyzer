
import os
import re
import sys
import subprocess

import streamlit as st
import spacy
import PyPDF2

# -------------------- OPENAI SETUP --------------------

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# -------------------- LOAD NLP MODEL --------------------

@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            with st.spinner("Installing the English language model..."):
                subprocess.run(
                    [
                        sys.executable, "-m", "spacy",
                        "download", "en_core_web_sm"
                    ],
                    check=True
                )
            return spacy.load("en_core_web_sm")
        except Exception as error:
            st.error(
                "Could not load the spaCy language model. "
                "Run: py -m spacy download en_core_web_sm"
            )
            st.stop()

nlp = load_nlp_model()

# -------------------- SKILL DATABASE --------------------

TECHNICAL_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "node.js", "express",
    "django", "flask", "fastapi", "sql", "postgresql", "mysql",
    "mongodb", "git", "github", "gitlab", "docker", "kubernetes",
    "aws", "azure", "gcp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "machine learning",
    "deep learning", "data structures", "algorithms",
    "rest api", "microservices", "ci/cd", "jenkins",
    "linux", "ubuntu", "windows", "excel", "power bi",
    "tableau", "spring boot", ".net", "php", "laravel",
    "ruby", "rails", "golang", "rust", "blockchain",
    "solidity", "vue", "svelte", "cybersecurity",
    "networking", "cloud computing", "devops", "agile",
    "scrum", "project management", "communication",
    "problem solving", "teamwork", "leadership",
    "data analysis", "data visualization", "statistics",
    "artificial intelligence", "generative ai", "nlp",
    "natural language processing", "powerpoint", "word",
    "manual testing", "automation testing", "selenium",
    "jira", "postman", "api testing", "software testing",
    "figma", "ui", "ux", "ui/ux"
]

# Match longer skills first to avoid missing multi-word skills.
TECHNICAL_SKILLS = sorted(
    set(TECHNICAL_SKILLS),
    key=len,
    reverse=True
)

# -------------------- TEXT EXTRACTION --------------------

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)


def extract_text_from_txt(uploaded_file):
    return uploaded_file.getvalue().decode(
        "utf-8", errors="ignore"
    )


def extract_resume_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    return extract_text_from_txt(uploaded_file)


# -------------------- SKILL EXTRACTION --------------------

def extract_skills(text):
    normalized_text = text.lower()
    found = set()

    # Match skill phrases in the complete text.
    for skill in TECHNICAL_SKILLS:
        pattern = (
            r"(?<![a-z0-9+#.])"
            + re.escape(skill)
            + r"(?![a-z0-9+#.])"
        )

        if re.search(pattern, normalized_text):
            found.add(skill)

    return found


# -------------------- SKILL GAP ANALYSIS --------------------

def analyze_skills(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched = resume_skills.intersection(job_skills)
    missing = job_skills.difference(resume_skills)

    if job_skills:
        match_pct = round(
            len(matched) / len(job_skills) * 100, 2
        )
    else:
        match_pct = 0.0

    return resume_skills, job_skills, matched, missing, match_pct


# -------------------- LEARNING RECOMMENDATIONS --------------------

def recommend_courses(missing_skills):
    courses = {
        "python": "https://www.coursera.org/specializations/python",
        "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "react": "https://react.dev/learn",
        "java": "https://dev.java/learn/",
        "sql": "https://www.w3schools.com/sql/",
        "mongodb": "https://learn.mongodb.com/",
        "aws": "https://aws.amazon.com/training/",
        "docker": "https://docs.docker.com/get-started/",
        "kubernetes": "https://kubernetes.io/docs/tutorials/",
        "machine learning": "https://www.coursera.org/specializations/machine-learning-introduction",
        "data structures": "https://www.geeksforgeeks.org/data-structures/",
        "cloud computing": "https://www.cloudskillsboost.google/",
        "html": "https://developer.mozilla.org/en-US/docs/Learn_web_development",
        "css": "https://developer.mozilla.org/en-US/docs/Web/CSS",
        "git": "https://git-scm.com/doc",
        "figma": "https://help.figma.com/",
        "selenium": "https://www.selenium.dev/documentation/",
        "postman": "https://learning.postman.com/",
        "software testing": "https://www.guru99.com/software-testing.html",
        "power bi": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/",
        "nlp": "https://www.nltk.org/",
        "generative ai": "https://www.cloudskillsboost.google/"
    }

    if not missing_skills:
        return "Excellent! No missing skills were detected from the listed skill database."

    recommendations = []

    for skill in sorted(missing_skills):
        url = courses.get(
            skill,
            "https://www.google.com/search?q="
            + skill.replace(" ", "+")
            + "+learning+resources"
        )

        recommendations.append(
            f"- **{skill.title()}**: {url}"
        )

    return "\n".join(recommendations)


# -------------------- CAREER REPORT --------------------

def generate_career_report(
    name, match_pct, matched, missing, use_openai
):
    if use_openai and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                client = OpenAI(api_key=api_key)

                prompt = f"""
You are a professional career advisor.

Analyze the following resume and job description skill comparison.

Candidate name: {name}
Skill match percentage: {match_pct}%
Matched skills: {', '.join(sorted(matched)) if matched else 'None'}
Missing skills: {', '.join(sorted(missing)) if missing else 'None'}

Write a personalized career improvement report in simple English.
Include:
1. Candidate strengths
2. Skills to improve
3. Suitable entry-level job roles
4. Three practical next steps

Keep the report under 200 words.
Do not invent qualifications or work experience.
Explain that the skill match percentage is an estimate.
"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful career advisor."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.5,
                    max_tokens=350
                )

                return response.choices[0].message.content

            except Exception as error:
                st.warning(
                    "OpenAI report generation failed. "
                    "Using the built-in report instead. "
                    f"Details: {error}"
                )

    # Built-in report works without an OpenAI API key.
    report = f"""
### Career Improvement Report for {name}

**Estimated skill match:** {match_pct}%

**Strengths**

Your resume contains the following skills that match the job
description: {', '.join(sorted(matched)) if matched else 'No matching skills detected'}.

**Areas for improvement**

Focus on learning: {', '.join(sorted(missing)) if missing else 'Continue improving your existing skills'}.

**Suggested entry-level job roles**

Depending on your education, projects, and experience, consider:
- Junior Software Developer
- Software Testing Trainee
- Junior Data Analyst
- Technical Support Associate

**Recommended next steps**

1. Learn the missing skills relevant to your target job.
2. Build small practical projects and add them to your portfolio.
3. Update your resume with relevant skills and genuine project experience.

**Important:** The skill match percentage is an estimate based on
the skills detected in the text. It is not a guarantee of job selection.
"""

    return report


# -------------------- APPLICATION HEADER --------------------

st.title("🤖 AI-Based Resume Analyzer")
st.subheader("Resume Analysis & Career Recommendation System")

st.write(
    "Upload a resume and enter a job description to identify "
    "matching skills, skill gaps, learning resources, and career advice."
)

st.divider()

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("⚙️ Settings")

    candidate_name = st.text_input(
        "Candidate name",
        value="Candidate"
    )

    use_openai = st.checkbox(
        "Generate report using OpenAI",
        value=False
    )

    if use_openai:
        if not OPENAI_AVAILABLE:
            st.warning("The OpenAI package is not installed.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning(
                "OPENAI_API_KEY is not configured. "
                "The built-in report will be used."
            )
        else:
            st.success("OpenAI API key detected.")

    st.caption(
        "OpenAI API usage may incur charges. "
        "A ChatGPT subscription does not automatically include API credits."
    )

# -------------------- INPUT SECTION --------------------

left, right = st.columns(2)

with left:
    st.markdown("### 📄 1. Upload Resume")

    resume_file = st.file_uploader(
        "Choose a resume in PDF or TXT format",
        type=["pdf", "txt"]
    )

with right:
    st.markdown("### 💼 2. Job Description")

    job_desc = st.text_area(
        "Paste the job description here",
        height=180,
        placeholder=(
            "Example: We are looking for a Python developer "
            "with SQL, Git, communication, and problem-solving skills."
        )
    )

st.caption(
    "Note: Image-only scanned PDFs may require OCR before their text can be analyzed."
)

# -------------------- ANALYZE BUTTON --------------------

if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):

    if resume_file is None:
        st.error("Please upload a resume first.")

    elif not job_desc.strip():
        st.error("Please enter a job description.")

    else:
        try:
            with st.spinner("Analyzing resume and matching skills..."):

                resume_text = extract_resume_text(resume_file)

                if not resume_text.strip():
                    st.error(
                        "No readable text was found. "
                        "Try a text-based PDF or TXT file."
                    )
                    st.stop()

                (
                    resume_skills,
                    job_skills,
                    matched,
                    missing,
                    match_pct
                ) = analyze_skills(resume_text, job_desc)

            st.success("Resume analysis completed!")

            # -------------------- SUMMARY METRICS --------------------

            st.markdown("## 📊 Analysis Summary")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Resume Skills", len(resume_skills))
            c2.metric("Required Skills", len(job_skills))
            c3.metric("Matched Skills", len(matched))
            c4.metric("Estimated Match", f"{match_pct}%")

            st.progress(int(match_pct))

            if not job_skills:
                st.warning(
                    "No skills from the built-in database were detected "
                    "in the job description. Try adding technical skills "
                    "or relevant keywords."
                )

            # -------------------- MATCHED AND MISSING SKILLS --------------------

            st.markdown("## 🎯 Skill Gap Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🟢 Matched Skills")

                if matched:
                    for skill in sorted(matched):
                        st.write(f"✅ {skill.title()}")
                else:
                    st.info("No matching skills were detected.")

            with col2:
                st.markdown("### 🔴 Missing Skills")

                if missing:
                    for skill in sorted(missing):
                        st.write(f"❌ {skill.title()}")
                else:
                    st.success(
                        "No missing skills were detected "
                        "from the listed skill database."
                    )

            # -------------------- ALL RESUME SKILLS --------------------

            with st.expander("View all skills detected in the resume"):
                if resume_skills:
                    st.write(", ".join(sorted(resume_skills)))
                else:
                    st.write("No skills detected.")

            # -------------------- LEARNING RESOURCES --------------------

            st.markdown("## 📚 Recommended Learning Resources")

            course_report = recommend_courses(missing)
            st.markdown(course_report)

            # -------------------- CAREER REPORT --------------------

            st.markdown("## 📝 Personalized Career Report")

            report = generate_career_report(
                candidate_name.strip() or "Candidate",
                match_pct,
                matched,
                missing,
                use_openai
            )

            st.markdown(report)

            # -------------------- DOWNLOAD REPORT --------------------

            report_text = f"""
AI RESUME ANALYZER REPORT

Candidate: {candidate_name or 'Candidate'}
Estimated Skill Match: {match_pct}%

Resume Skills:
{', '.join(sorted(resume_skills)) or 'None detected'}

Job Skills:
{', '.join(sorted(job_skills)) or 'None detected'}

Matched Skills:
{', '.join(sorted(matched)) or 'None'}

Missing Skills:
{', '.join(sorted(missing)) or 'None'}

LEARNING RESOURCES:
{course_report}

CAREER REPORT:
{report}

Note: The skill match is an estimate based on detected keywords.
"""

            st.download_button(
                label="⬇️ Download Analysis Report",
                data=report_text,
                file_name="resume_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:
            st.error(
                "An error occurred while analyzing the resume. "
                "Please check the uploaded file and try again."
            )
            st.exception(error)

else:
    st.info(
        "Upload a resume, paste a job description, "
        "and click Analyze Resume to begin."
    )

# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "MCA Practical Project | AI-Based Resume Analyzer & "
    "Career Recommendation System"
)
