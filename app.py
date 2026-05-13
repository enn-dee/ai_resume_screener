import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_extractor import extract_text_from_pdf
from utils.text_cleaner import clean_text
from utils.skill_extractor import extract_skills

model = SentenceTransformer('all-MiniLM-L6-v2')

job_description = """
Looking for a Python developer with:
- Flask
- REST APIs
- SQL
- Backend development
- Git experience
"""

cleaned_job = clean_text(job_description)

job_skills = extract_skills(cleaned_job)

# Generate JD embedding
job_embedding = model.encode([cleaned_job])

resume_folder = "resumes"

results = []

for file_name in os.listdir(resume_folder):

    if file_name.endswith(".pdf"):

        pdf_path = os.path.join(resume_folder, file_name)

        print(f"\nProcessing: {file_name}")

        resume_text = extract_text_from_pdf(pdf_path)

        cleaned_resume = clean_text(resume_text)

        resume_skills = extract_skills(cleaned_resume)

        # Generate embedding
        resume_embedding = model.encode([cleaned_resume])

        similarity = cosine_similarity(
            job_embedding,
            resume_embedding
        )

        semantic_score = similarity[0][0] * 100

        matched_skills = set(job_skills).intersection(
            set(resume_skills)
        )

        skill_match_score = (
            len(matched_skills) / len(job_skills)
        ) * 100 if job_skills else 0

        final_score = (
            (0.7 * semantic_score)
            +
            (0.3 * skill_match_score)
        )

        results.append({
            "resume": file_name,
            "semantic_score": semantic_score,
            "skill_score": skill_match_score,
            "final_score": final_score,
            "skills": resume_skills,
            "matched_skills": list(matched_skills)
        })

ranked_results = sorted(
    results,
    key=lambda x: x["final_score"],
    reverse=True
)

print("\n===== RANKED CANDIDATES =====\n")

for rank, result in enumerate(ranked_results, start=1):

    print(f"{rank}. {result['resume']}")

    print(
        f"Semantic Score: "
        f"{result['semantic_score']:.2f}%"
    )

    print(
        f"Skill Score: "
        f"{result['skill_score']:.2f}%"
    )

    print(
        f"Final Score: "
        f"{result['final_score']:.2f}%"
    )

    print(f"Skills Found: {result['skills']}")

    print(
        f"Matched Skills: "
        f"{result['matched_skills']}"
    )

    print("-" * 50)