import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_extractor import extract_text_from_pdf
from utils.text_cleaner import clean_text
from utils.skill_extractor import extract_skills

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "AI Resume Screener API Running"


@app.route("/analyze", methods=["POST"])
def analyze_resumes():

    job_description = request.form.get(
        "job_description"
    )

    if not job_description:
        return jsonify({
            "error": "Job description missing"
        }), 400

    uploaded_files = request.files.getlist(
        "resumes"
    )
    if len(uploaded_files) == 0:
        return jsonify({
            "error": "No resumes uploaded"
        }), 400

    # Clean job description
    cleaned_job = clean_text(job_description)

    job_skills = extract_skills(cleaned_job)

    # Generate job embedding
    job_embedding = model.encode([cleaned_job])

    results = []

    for file in uploaded_files:

        if file.filename.endswith(".pdf"):

            # Secure filename
            filename = secure_filename(
                file.filename
            )

            save_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(save_path)

            resume_text = extract_text_from_pdf(
                save_path
            )

            cleaned_resume = clean_text(
                resume_text
            )

            resume_skills = extract_skills(
                cleaned_resume
            )

            # Generate resume embedding
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

                "resume": filename,

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