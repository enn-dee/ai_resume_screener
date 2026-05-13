from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_extractor import extract_text_from_pdf
from utils.text_cleaner import clean_text

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Job description
job_description = """
Looking for a Python developer with Flask,
REST API, SQL, and backend development experience.
"""

# Extract PDF text
resume_text = extract_text_from_pdf("resumes/resume.pdf")

# Clean text
cleaned_resume = clean_text(resume_text)
cleaned_job = clean_text(job_description)

print("\n===== CLEANED RESUME =====\n")
print(cleaned_resume)

# Embeddings
job_embedding = model.encode([cleaned_job])
resume_embedding = model.encode([cleaned_resume])

# Similarity
similarity = cosine_similarity(job_embedding, resume_embedding)

match_percentage = similarity[0][0] * 100

print(f"\nMatch Score: {match_percentage:.2f}%")