import streamlit as st
from recommender import recommend
import pandas as pd

movies = pd.read_csv("dataset/movies.csv")

st.title("Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        with cols[i % 5]:
            st.write(movie)