from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_FILE = Path("spam_model.pkl")
VECTORIZER_FILE = Path("vectorizer.pkl")

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

@app.route("/", methods=["GET", "POST"])
def home():
    email_text = ""
    result = None
    confidence = None
    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if email_text:
            email_vector = vectorizer.transform([email_text])
            prediction = model.predict(email_vector)[0]
            probabilities = model.predict_proba(email_vector)[0]
            confidence = round(float(max(probabilities)) * 100)
            result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", email_text=email_text,
                           result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
