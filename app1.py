import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
import base64
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure stopwords are downloaded only once
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_dir, "corpora/stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Load stopwords
@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('logistic_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Twitter_Data.csv")

# Convert image to base64
def get_base64_bg(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Preprocess and predict sentiment
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

# Display sentiment result
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

# Main App
def main():
    st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

    # Set background
    bg_image = get_base64_bg("image_n.jpg")  # Update this if you use another image

    # Transparent inputs, no white shades, adaptive dark text
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

        textarea, input[type="text"], div[data-baseweb="select"] {{
            background-color: transparent !important;
            color: #000000 !important;
            border: 2px solid #444 !important;
            font-weight: 600;
            border-radius: 8px;
            box-shadow: 1px 1px 4px rgba(0,0,0,0.3);
        }}

        label, p, h1, h2, h3, h4, h5 {{
            color: #111111 !important;
            text-shadow: 0px 0px 3px rgba(255,255,255,0.6);
        }}

        .stButton > button {{
            background-color: #fff !important;
            color: #111 !important;
            border: 1px solid #222 !important;
            font-weight: 600;
            border-radius: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📊 Public Comment Sentiment Analysis")

    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    try:
        df = load_data()
        if 'clean_text' not in df.columns:
            st.error("Column 'clean_text' not found in Twitter_Data.csv.")
            return

        tweet_list = df['clean_text'].dropna().unique().tolist()[:1000]

        with st.form("sentiment_form"):
            selected_text = st.selectbox("Choose a sample text to analyze", tweet_list)
            st.markdown("Or enter your own text:")
            text_input = st.text_area("Custom text to analyze", height=150)
            submitted = st.form_submit_button("Analyze")

        if submitted:
            input_text = text_input.strip() if text_input.strip() else selected_text
            sentiment = predict_sentiment(input_text, model, vectorizer, stop_words)
            st.markdown(create_card(input_text, sentiment), unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("🛑 'Twitter_Data.csv' not found. Please make sure it's in the app folder.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
