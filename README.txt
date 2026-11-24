EMAIL SPAM CLASSIFIER - MACHINE LEARNING VERSION

WHAT THIS PROJECT DOES
A small website where a user pastes an email and receives a Spam / Not Spam
prediction. It uses a Multinomial Naive Bayes model and TF-IDF text features.

TECHNOLOGIES
- Python
- Flask (website backend)
- HTML and CSS (website design)
- scikit-learn (machine learning)
- TF-IDF Vectorizer
- Multinomial Naive Bayes

HOW TO RUN (Windows)
1. Install Python 3.10 or newer.
2. Extract this ZIP folder.
3. Open Command Prompt or Terminal in the extracted project folder.
4. Optional but recommended: create a virtual environment:
      python -m venv venv
   Activate it:
      venv\Scripts\activate
5. Install the libraries:
      pip install -r requirements.txt
6. Train the model:
      python train_model.py
   This creates spam_model.pkl and vectorizer.pkl.
7. Start the website:
      python app.py
8. Open the local address shown in the terminal, usually:
      http://127.0.0.1:5000

IMPORTANT
- Run train_model.py once before app.py.
- This project uses a small sample dataset included as emails.csv.
- The model is for learning and demonstration only. The sample dataset is
  small and does not represent all real-world emails.
- Confidence is the model's predicted probability, not a guarantee of safety.
- Do not paste private emails, passwords, OTPs, or confidential information.

HOW TO EXPLAIN IT IN A VIVA
1. The user enters an email message on the website.
2. Flask receives the text from the form.
3. TF-IDF converts the words into numerical features.
4. Multinomial Naive Bayes uses patterns learned from labelled examples.
5. The website displays Spam or Not Spam with a confidence estimate.

PROJECT FLOW
Email text -> TF-IDF Vectorizer -> Naive Bayes Model -> Prediction -> Website result
