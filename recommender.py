import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/movies.csv")

movies['genres'] = movies['genres'].str.replace('|',' ')

cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['genres'])

similarity = cosine_similarity(count_matrix)

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)

    return recommended