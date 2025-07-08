import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text
import joblib

# Load models
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

nb_model = joblib.load('model/nb_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
lstm_model = tf.keras.models.load_model('model/fake_news_lstm_model.h5')

max_len = 300

def predict_lstm(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = lstm_model.predict(padded)[0][0]
    return ("REAL", round(pred * 100, 2)) if pred >= 0.5 else ("FAKE", round((1 - pred) * 100, 2))

def predict_nb(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = nb_model.predict(vec)[0]
    prob = nb_model.predict_proba(vec)[0][pred]
    return ("REAL", round(prob * 100, 2)) if pred == 1 else ("FAKE", round(prob * 100, 2))

# 🧑‍🎨 CSS to remove Streamlit padding and center content
st.set_page_config(page_title="Fake News Classifier", layout="centered")

st.markdown("""
    <style>
        /* Remove default padding and white box */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .main-title {
            font-size: 36px;
            font-weight: 800;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .stButton button {
            background-color: #27ae60;
            color: white;
            font-size: 16px;
            padding: 10px 30px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# 🌟 UI Layout
st.markdown('<div class="main-title">🧠 Fake News Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classify news as <strong>REAL</strong> or <strong>FAKE</strong> using NLP models</div>', unsafe_allow_html=True)

model_option = st.radio("📊 Choose a model", ["Naive Bayes (TF-IDF)", "LSTM (Deep Learning)"])
user_input = st.text_area("📝 Enter the news content below", height=200)

if st.button("🚀 Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        if model_option == "Naive Bayes (TF-IDF)":
            label, confidence = predict_nb(user_input)
        else:
            label, confidence = predict_lstm(user_input)

        if label == "REAL":
            st.success(f"✅ REAL NEWS — Confidence: {confidence}%")
        else:
            st.error(f"❌ FAKE NEWS — Confidence: {confidence}%")
