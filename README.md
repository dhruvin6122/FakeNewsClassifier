# 📰 Fake News Detection using NLP, TF-IDF, Word2Vec, Naive Bayes & LSTM

This project helps users identify whether a news article is **REAL** or **FAKE** using **two powerful models**:
- 🔹 Traditional Machine Learning (Naive Bayes + TF-IDF)
- 🔹 Deep Learning (LSTM + Word2Vec)

Deployed with **Streamlit** and built for production-ready demos.

---

## 🚀 Demo

## 📸 Screenshots

Visual demonstration of model predictions:

---

### 🔍 LSTM Model Predictions

| ❌ Fake News | ✅ Real News |
|-------------|-------------|
| ![LSTM Fake](https://raw.githubusercontent.com/dhruvin6122/fake-news-detector/main/FakeNewsClassifier/screenshorts/Lstm_Fake1.png) | ![LSTM Real](https://raw.githubusercontent.com/dhruvin6122/fake-news-detector/main/FakeNewsClassifier/screenshorts/Lstm_True1.png) |

---

### 🤖 Naive Bayes Predictions

| ❌ Fake News | ✅ Real News |
|-------------|-------------|
| ![Naive Fake](https://raw.githubusercontent.com/dhruvin6122/fake-news-detector/main/FakeNewsClassifier/screenshorts/Nave_Fake1.png) | ![Naive Real](https://raw.githubusercontent.com/dhruvin6122/fake-news-detector/main/FakeNewsClassifier/screenshorts/Nave_True1.png) |


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

## 🔮 Future Enhancements

Here are some exciting features planned for upcoming versions:

- 🚀 **Transformer Integration**: Upgrade LSTM with BERT, RoBERTa, or DistilBERT for better contextual understanding.
- 📰 **Live News Testing**: Add a News API (e.g., NewsAPI.org) to fetch and classify trending news automatically.
- 🔍 **Explainable AI**: Use SHAP or LIME to explain why a model predicted news as fake/real.
- 📊 **Prediction Dashboard**: Store past predictions and display user statistics.
- ☁️ **Cloud Deployment**: Deploy on Streamlit Cloud, Hugging Face Spaces, or Render.
- 🔒 **Authentication**: Add user login system to restrict access or allow personal saved results.
- 🧪 **Model Comparison Mode**: Visualize accuracy/F1 score differences across models.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, distribute, and modify the code with proper attribution.

---

## 📦 Download Whole Project With Models And Datasets From Google Drive

You can download the pre-trained models (LSTM + Naive Bayes) And Dataset here:

🔗 [📁 Google Drive Folder – Download Models](https://drive.google.com/drive/folders/1vkZuEqSc27pHc9xOxNlvPzKx8J7p9xWE?usp=sharing)

> Includes:
> - `DataSets`
> - `fake_news_lstm_model.h5`
> - `tokenizer.pkl`
> - `naive_bayes_model.pkl`
> - `vectorizer.pkl`

---

## 🙋‍♂️ Author

**Dhruvin Patel**  
🎓 B.Tech – Computer Science  
📍 Gujarat, India  
📧 Email: pateldhruvin6122@gmail.com  
🔗 GitHub: [@dhruvin6122](https://github.com/dhruvin6122)    
💼 LinkedIn: [View Profile](https://www.linkedin.com/in/patel-dhruvin-70b602280?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)



