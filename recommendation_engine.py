from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="SHL Assessment Recommendation API")

# SHL Assessment Catalogue
assessments = {
    "assessment_name": [
        "Python Programming Test",
        "Data Analysis Test",
        "Cognitive Ability Test",
        "Communication Skills Test"
    ],
    "description": [
        "Python coding, algorithms, machine learning",
        "Data analysis, SQL, statistics",
        "Logical reasoning, analytical thinking",
        "Grammar, verbal communication"
    ],
    "url": [
        "https://www.shl.com/",
        "https://www.shl.com/",
        "https://www.shl.com/",
        "https://www.shl.com/"
    ]
}

df = pd.DataFrame(assessments)

class JobInput(BaseModel):
    job_description: str

@app.post("/recommend")
def recommend_assessments(data: JobInput):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(
        df["description"].tolist() + [data.job_description]
    )

    similarity_scores = cosine_similarity(
        vectors[-1], vectors[:-1]
    ).flatten()

    df["score"] = similarity_scores
    result = df.sort_values(by="score", ascending=False)

    return result[["assessment_name", "url", "score"]].head(3).to_dict(orient="records")

