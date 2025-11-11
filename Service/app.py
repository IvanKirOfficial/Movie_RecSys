import os
import pickle
import sys

import streamlit as st
import requests
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ML.Models.Recommendations import get_recommendations

global needVPN

def fetch_poster(movie_id):
    global needVPN
    needVPN = False

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=12cff3c8dd61f25fc4fda607fa5309e5&language=en-US"
    try:
        data = requests.get(url)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
    except requests.exceptions.RequestException as _:
        needVPN = True

    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"


def recommend(title):
    recommendations_list = get_recommendations(title, movies, cosine_sim)
    recommended_movie_names_list = []
    recommended_movie_posters_list = []
    recommended_movie_years_list = []
    recommended_movie_ratings_list = []

    for title in recommendations_list[:5]:
        row_idx = movies.index[movies['title'] == title][0]

        movie_id = movies.loc[row_idx]['id']

        recommended_movie_posters_list.append(fetch_poster(movie_id))
        recommended_movie_names_list.append(movies.loc[row_idx]['title'])
        recommended_movie_years_list.append(movies.loc[row_idx]['year'])
        recommended_movie_ratings_list.append(movies.loc[row_idx]['vote_average'])

    if needVPN:
        st.error(f"Error fetching poster: Try to use VPN")

    return recommended_movie_names_list, recommended_movie_posters_list, recommended_movie_years_list, recommended_movie_ratings_list


st.set_page_config(layout="wide")
st.header('Movie Recommender System')

try:
    movies_list = pickle.load(open('ML/Data/Processed/movie_list.pkl', 'rb'))
    movies = pd.DataFrame(movies_list)
    cosine_sim = pickle.load(open('ML/Data/Processed/cosine_sim.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the data processing notebook first.")
    st.stop()

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    with st.spinner('Finding recommendations...'):
        recommended_movie_names, recommended_movie_posters, recommended_movie_years, recommended_movie_ratings = recommend(selected_movie)

    if recommended_movie_names:
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(recommended_movie_names[i])
                st.image(recommended_movie_posters[i])
                year = recommended_movie_years[i]
                if pd.notna(year):
                    st.caption(f"Year: {int(year)}")
                else:
                    st.caption("Year: N/A")

                rating = recommended_movie_ratings[i]
                st.caption(f"Rating: {rating:.1f} ⭐")