import threading
import streamlit as st
import requests
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# ---------------- FASTAPI BACKEND ---------------- #

api = FastAPI(title="SHL Assessment Recommendation API")

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

@api.post("/recommend")
def recommend_assessments(data: JobInput):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(
        df["description"].tolist() + [data.job_description]
    )

    similarity_scores = cosine_similarity(
        vectors[-1], vectors[:-1]
    ).flatten()

    df["score"] = similarity_scores
    result = df.sort_values(by="score", ascending=False).head(3)

    return result[["assessment_name", "url", "score"]].to_dict(orient="records")


# ---------------- STREAMLIT FRONTEND ---------------- #

def run_streamlit():
    st.set_page_config(page_title="SHL Assessment Recommender")
    st.title("SHL Assessment Recommendation System")

    job_description = st.text_area(
        "Enter Job Description",
        placeholder="Looking for a Python developer with data analysis skills"
    )

    if st.button("Recommend Assessments"):
        if job_description.strip():
            response = requests.post(
                "http://127.0.0.1:8000/recommend",
                json={"job_description": job_description}
            )

            if response.status_code == 200:
                st.subheader("Recommended Assessments")
                for r in response.json():
                    st.write(f"**{r['assessment_name']}**")
                    st.write(f"Score: {round(r['score'], 2)}")
                    st.write(f"[SHL Link]({r['url']})")
                    st.divider()
            else:
                st.error("API error occurred")
        else:
            st.warning("Please enter a job description")


# ---------------- MAIN ---------------- #

def run_api():
    uvicorn.run(api, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    # Run FastAPI in background
    threading.Thread(target=run_api, daemon=True).start()

    # Run Streamlit UI
    run_streamlit()

