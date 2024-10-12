import streamlit as st
import pickle
import requests
import nltk
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

from nltk.stem import PorterStemmer

ps = PorterStemmer()

st.header('Movie Recommender System')



input_sms = st.text_area("Let me know which type of movie you want to see:")


movies = pickle.load(open('movie_list.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))


def transform_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)


def filter_input(text):
    text = transform_data(text)
    input_vector = tfidf.transform([text]).toarray()
    return input_vector


def recommend(inp_vector, vector):
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(inp_vector, vector)
    sorted_indices = similarity_scores.argsort()[0][::-1]

    # Get top 5 similar movies
    top_similar_movies_indices = sorted_indices[:5]
    top_movies = movies.iloc[top_similar_movies_indices]

    recommended_movie_names = []
    recommended_movie_posters = []

    for movie in top_movies.itertuples():
        recommended_movie_names.append(movie.title)
        recommended_movie_posters.append(fetch_poster(movie.id))

    return recommended_movie_names, recommended_movie_posters


def fetch_poster(movie_id):
    # you have to put your tmdb api key here
    url = "https://api.themoviedb.org/3/movie/{}?api_key=<your-tmdb-api-key>&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data.get('poster_path', None)
    if poster_path:
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    return None  # Return None if no poster found


# Recommending movies based on input
if st.button('Show Recommendation'):
    text_vector = filter_input(input_sms)
    recommended_movie_names, recommended_movie_posters = recommend(text_vector, vector)

    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(5):
        with eval(f'col{i + 1}'):
            st.text(recommended_movie_names[i])
            if recommended_movie_posters[i]:
                st.image(recommended_movie_posters[i])

