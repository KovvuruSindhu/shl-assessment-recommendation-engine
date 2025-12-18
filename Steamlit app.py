import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Recommender")

st.title("SHL Assessment Recommendation System")

job_description = st.text_area(
    "Enter Job Description",
    placeholder="e.g. Looking for a Python developer with data analysis experience"
)

if st.button("Recommend Assessments"):
    if job_description.strip():
        response = requests.post(
            "http://127.0.0.1:8000/recommend",
            json={"job_description": job_description}
        )

        if response.status_code == 200:
            results = response.json()
            st.subheader("Recommended Assessments")
            for r in results:
                st.write(f"**{r['assessment_name']}**")
                st.write(f"Score: {round(r['score'], 2)}")
                st.write(f"[SHL Link]({r['url']})")
                st.divider()
        else:
            st.error("API error occurred")
    else:
        st.warning("Please enter a job description")
