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

## 🔮 Future Enhancements

Here are some exciting features planned for upcoming versions:

- 🚀 **Transformer Integration**: Upgrade LSTM with BERT, RoBERTa, or DistilBERT for better contextual understanding.
- 📰 **Live News Testing**: Add a News API (e.g., NewsAPI.org) to fetch and classify trending news automatically.
- 🔍 **Explainable AI**: Use SHAP or LIME to explain why a model predicted news as fake/real.
- 📊 **Prediction Dashboard**: Store past predictions and display user statistics.
- ☁️ **Cloud Deployment**: Deploy on Streamlit Cloud, Hugging Face Spaces, or Render.
- 🔒 **Authentication**: Add user login system to restrict access or allow personal saved results.
- 🧪 **Model Comparison Mode**: Visualize accuracy/F1 score differences across models.

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, distribute, and modify the code with proper attribution.

## 🙋‍♂️ Author

**Dhruvin Patel**  
🎓 B.Tech – Computer Science  
📍 Gujarat, India  
📧 Email: pateldhruvin6122@gmail.com  
🔗 GitHub: [@dhruvin6122](https://github.com/dhruvin6122)    
💼 LinkedIn: [View Profile]([https://linkedin.com/](https://www.linkedin.com/in/patel-dhruvin-70b602280?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app))


