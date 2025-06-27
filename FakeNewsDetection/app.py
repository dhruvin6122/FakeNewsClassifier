from flask import Flask, render_template, request
import joblib
from clean_text import clean_text

# Load model and vectorizer
model = joblib.load("models/naive_bayes_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        title = request.form["title"]
        text = request.form["text"]
        combined = title + " " + text
        cleaned = clean_text(combined)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]

        prediction = "REAL" if pred == 1 else "FAKE"
        probability = round(prob * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
