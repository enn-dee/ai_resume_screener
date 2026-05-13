from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample Job Description
job_description = """
Looking for a Python developer with experience in:
- JAVA
- REST APIs
- SQL
- Backend Development
"""

# Sample Resume
resume = """
Software engineer experienced in Python backend systems,
Flask applications, API development, and database management.
"""

# Convert text into embeddings
job_embedding = model.encode([job_description])
resume_embedding = model.encode([resume])

# Compare similarity
similarity_score = cosine_similarity(job_embedding, resume_embedding)

# Convert to percentage
match_percentage = similarity_score[0][0] * 100

print(f"Match Score: {match_percentage:.2f}%")