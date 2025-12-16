import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# SHL Assessment Catalogue
assessments = {
"assessment_name": [
"Python Programming Test",
"Data Analysis Test",
"Cognitive Ability Test",
"Communication Skills Test"
],
"description": [
"Python coding, data structures, algorithms",
"Data analysis, SQL, statistics, visualization",
"Logical reasoning, analytical thinking, problem solving",
"Grammar, verbal ability, communication skills"
]
}


df = pd.DataFrame(assessments)


# Input Job Description (User Input)
job_description = input("Enter the job description:")


'''Looking for a research intern with strong Python, data analysis,
machine learning, and problem-solving skills.'''
# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['description'].tolist() + [job_description])


# Cosine Similarity
similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()


# Ranking Results
df['similarity_score'] = similarity_scores
recommendations = df.sort_values(by='similarity_score', ascending=False)


print(recommendations[['assessment_name', 'similarity_score']])
