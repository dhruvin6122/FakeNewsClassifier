# 📰 Fake News Detection using NLP, TF-IDF, Word2Vec, Naive Bayes & LSTM

This project helps users identify whether a news article is **REAL** or **FAKE** using **two powerful models**:
- 🔹 Traditional Machine Learning (Naive Bayes + TF-IDF)
- 🔹 Deep Learning (LSTM + Word2Vec)

Deployed with **Streamlit** and built for production-ready demos.

---

## 🚀 Demo

![App Screenshot](screenshots/demo.png)

---

## 🔍 Features

- 🔀 Choose between **Naive Bayes** or **LSTM**
- ✍️ Real-time news text prediction
- 🧠 Pretrained tokenizer and embedding matrix
- 🔠 Cleaned with POS-tagging + Lemmatization
- ✅ Shows confidence score of prediction
- 💻 White & clean professional UI

---

## 🧠 Models

### ✅ Naive Bayes + TF-IDF
- Uses `TfidfVectorizer` with `max_features=5000`
- Trained with `MultinomialNB`
- Fast and interpretable

### 🔁 LSTM + Word2Vec
- Word embeddings trained via `gensim.Word2Vec`
- Sequence padded to length 300
- `Keras` based LSTM model
- Better contextual understanding

---

## 🛠️ Tech Stack

| Layer         | Tools / Libraries                           |
|---------------|----------------------------------------------|
| Language      | Python 3.10                                  |
| UI            | Streamlit                                    |
| NLP           | NLTK, Gensim                                 |
| ML/DL         | Scikit-learn, TensorFlow, Keras              |
| Vectorization | TF-IDF, Word2Vec                             |
| Model Storage | Pickle, Joblib                               |

---

## 📁 Folder Structure

