# AI Resume Screener

An AI-powered Resume Screening System built using Python, NLP, Sentence Transformers, and Flask.

This project automates resume screening by:
- extracting text from resume PDFs
- analyzing candidate skills
- comparing resumes with job descriptions
- ranking candidates using AI-based semantic similarity

---

# Features

- Resume PDF upload API
- Job description matching
- Skill extraction
- Semantic similarity scoring
- Candidate ranking
- Flask backend API
- Multi-resume support

---

# Tech Stack

## Backend
- Python
- Flask

## AI / NLP
- Sentence Transformers
- Scikit-learn
- NLP preprocessing

## Other Tools
- PyPDF2
- NumPy

---

# Project Structure

```text
ai-resume-screener/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── uploads/
│
├── utils/
│   ├── pdf_extractor.py
│   ├── text_cleaner.py
│   ├── skill_extractor.py
│   └── skills.py
│
└── README.md
```

---

# How It Works

```text
User Uploads Resumes + Job Description
                ↓
         Flask Backend API
                ↓
         PDF Text Extraction
                ↓
          Text Cleaning
                ↓
          Skill Extraction
                ↓
      Sentence Transformer Embeddings
                ↓
        Semantic Similarity Analysis
                ↓
         Candidate Ranking
                ↓
            JSON Response
```

---

# Installation

## 1. Clone Repository

```bash
git clone git@github.com:enn-dee/ai_resume_screener.git

cd ai-resume-screener
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
python app.py
```

Server starts at:

```text
http://127.0.0.1:5000
```

---

# API Endpoint

## Analyze Resumes

### Endpoint

```http
POST /analyze
```

---

### Form Data

| Key | Type |
|---|---|
| job_description | Text |
| resumes | File (multiple PDFs supported) |

---

# Example Response

```json
[
  {
    "resume": "resume1.pdf",
    "semantic_score": 88.5,
    "skill_score": 75.0,
    "final_score": 84.45,
    "skills": [
      "python",
      "flask",
      "sql"
    ],
    "matched_skills": [
      "python",
      "sql"
    ]
  }
]
```

---

# AI Concepts Used

- NLP preprocessing
- Semantic similarity
- Sentence embeddings
- Skill matching
- Resume ranking
- Transformer models

---

# Future Improvements

- Database integration
- Authentication system
- Admin dashboard
- ATS compatibility scoring
- Advanced NLP models
- Cloud deployment
- Docker support

---

# Learning Outcomes

This project helps in understanding:
- Flask backend APIs
- NLP fundamentals
- Sentence embeddings
- Semantic similarity systems
- AI backend workflows
- File upload handling
- Resume ranking systems

---

# License

This project is for learning and educational purposes.