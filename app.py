import os

from flask import Flask, jsonify

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_extractor import extract_text_from_pdf
from utils.text_cleaner import clean_text
from utils.skill_extractor import extract_skills

from flask import Flask
model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Resume Screener API Running"

@app.route("/analyze")
def analyze_resumes():

    job_description = """
    Looking for a Python developer with:
    Flask, SQL, REST APIs, Git
    """

    cleaned_job = clean_text(job_description)

    job_skills = extract_skills(cleaned_job)

    job_embedding = model.encode([cleaned_job])

    resume_folder = "resumes"

    results = []

    for file_name in os.listdir(resume_folder):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(
                resume_folder,
                file_name
            )

            resume_text = extract_text_from_pdf(
                pdf_path
            )

            cleaned_resume = clean_text(
                resume_text
            )

            resume_skills = extract_skills(
                cleaned_resume
            )

            resume_embedding = model.encode(
                [cleaned_resume]
            )

            similarity = cosine_similarity(
                job_embedding,
                resume_embedding
            )

            semantic_score = (
                similarity[0][0] * 100
            )

            matched_skills = set(
                job_skills
            ).intersection(set(resume_skills))

            skill_match_score = (
                len(matched_skills)
                /
                len(job_skills)
            ) * 100 if job_skills else 0

            final_score = (
                (0.7 * semantic_score)
                +
                (0.3 * skill_match_score)
            )

            results.append({
    "resume": file_name,

    "semantic_score": float(
        round(semantic_score, 2)
    ),

    "skill_score": float(
        round(skill_match_score, 2)
    ),

    "final_score": float(
        round(final_score, 2)
    ),

    "skills": resume_skills,

    "matched_skills": list(
        matched_skills
    )
})

    ranked_results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return jsonify(ranked_results)

if __name__ == "__main__":
    app.run(debug=True)