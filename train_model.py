"""
Train a small Naive Bayes model for the Email Spam Classifier.
This is a beginner demonstration dataset, not a production email filter.
"""
import csv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = Path("emails.csv")

texts, labels = [], []
with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        texts.append(row["text"])
        labels.append(int(row["label"]))

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vector, y_train)

predictions = model.predict(X_test_vector)
print("Model accuracy on this small test split:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, target_names=["Not Spam", "Spam"], zero_division=0))

# Save the trained model and the matching text vectorizer.
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved spam_model.pkl and vectorizer.pkl")
