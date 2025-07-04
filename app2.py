import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
import base64
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_dir, "corpora/stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_dir)

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

@st.cache_resource
def load_model_and_vectorizer():
    with open('logistic_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("Twitter_Data.csv")

def get_base64_bg(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)[0]
    return { -1: "Negative", 0: "Neutral", 1: "Positive" }.get(sentiment, "Unknown")

def create_card(tweet_text, sentiment):
    color = {"Positive": "#4CAF50", "Neutral": "#999999", "Negative": "#f44336"}.get(sentiment, "#808080")
    icon = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}.get(sentiment, "")
    card_html = f"""
    <div style="background-color: {color}; padding: 16px; border-radius: 10px; margin: 15px 0;">
        <h5 style="color: white; margin: 0;">{icon} {sentiment} Sentiment</h5>
        <p style="color: white; font-size: 16px;">{tweet_text}</p>
    </div>
    """
    return card_html

def main():
    st.set_page_config(page_title="Public Sentiment Analysis", layout="wide")

    bg_image = get_base64_bg("image_n.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }}
        h1, h5, p, label {{
            color: white !important;
            text-shadow: 1px 1px 2px black;
        }}
        .stTextInput > div > input, 
        .stTextArea > div > textarea {{
            background: transparent !important;
            color: white !important;
            border: 2px solid white;
        }}
        .stSelectbox > div > div {{
            background: transparent !important;
            color: white !important;
            border: 2px solid white;
        }}
        .stButton > button {{
            background-color: white;
            color: black;
            font-weight: bold;
            border-radius: 5px;
            margin-top: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📊 Public Comment Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    try:
        df = load_data()
        if 'clean_text' not in df.columns:
            st.error("Column 'clean_text' not found in Twitter_Data.csv.")
            return

        tweet_list = df['clean_text'].dropna().unique().tolist()[:1000]

        with st.form("sentiment_form"):
            selected_text = st.selectbox("Choose a sample text to analyse", tweet_list)
            st.markdown("Or choose your own text:")
            text_input = st.text_area("Custom text to analyse", height=150)
            submitted = st.form_submit_button("Analyze")

        if submitted:
            input_text = text_input.strip() if text_input.strip() else selected_text
            sentiment = predict_sentiment(input_text, model, vectorizer, stop_words)
            st.markdown(create_card(input_text, sentiment), unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("🛑 'Twitter_Data.csv' not found.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
