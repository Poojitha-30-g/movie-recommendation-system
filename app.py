import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")

# Convert text to numbers
cv = CountVectorizer()
matrix = cv.fit_transform(movies['genre'])

# Find similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    result = []
    for i in movie_list:
        result.append(movies.iloc[i[0]].title)
    return result

# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    for movie in recommendations:
        st.write(movie)
