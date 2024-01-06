import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to load the model and preprocess data
@st.cache(allow_output_mutation=True)
def load_model():
    with open('book_recommendation_model.pkl', 'rb') as model_file:
        model_data = pickle.load(model_file)

    df = model_data['df']
    tfidf_vectorizer = model_data['tfidf_vectorizer']
    tfidf_matrix = model_data['tfidf_matrix']
    cosine_sim = model_data['cosine_sim']

    return df, tfidf_vectorizer, tfidf_matrix, cosine_sim

# Load the model and data
loaded_df, loaded_tfidf_vectorizer, loaded_tfidf_matrix, loaded_cosine_sim = load_model()

# Function to preprocess text
@st.cache_data
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Function to recommend books
@st.cache_data
def recommend_books(query, cosine_sim=loaded_cosine_sim, df=loaded_df, vectorizer=loaded_tfidf_vectorizer):
    processed_query = preprocess_text(query)
    query_df = pd.DataFrame({'Combined_Features': [processed_query]})
    query_vector = vectorizer.transform(query_df['Combined_Features'])
    sim_scores = linear_kernel(query_vector, loaded_tfidf_matrix).flatten()
    book_indices = sim_scores.argsort()[::-1][:10]
    return df['Title'].iloc[book_indices]

# Streamlit app
st.title("Books Search Engine")
st.header("Web Mining IS684")

# Input field for user to enter a specific book title
specific_book_title = st.text_input("Enter a specific book title:", "")

# Button to trigger book recommendation for the specific title
if st.button("Recommend Books for Specific Title"):
    if specific_book_title:
        recommended_books = recommend_books(specific_book_title)
        st.subheader("Recommended Books:")
        st.write(recommended_books)
    else:
        st.warning("Please enter a specific book title.")
