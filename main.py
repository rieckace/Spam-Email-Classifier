# app.py

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# Load and prepare dataset
# -------------------------------
@st.cache_resource
def load_model():
    file_path = "emails.csv"   # make sure emails.csv is in the same folder
    df = pd.read_csv(file_path)

    # Dataset is in bag-of-words format → convert to pseudo-text
    X = df.drop(columns=["Email No.", "Prediction"])
    y = df["Prediction"].astype(int)

    texts = []
    for _, row in X.iterrows():
        words = []
        for word, count in row.items():
            words.extend([word] * int(count))
        texts.append(" ".join(words))

    # Vectorize text
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(texts)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42
    )

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, acc

model, vectorizer, acc = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📧 Naive Bayes Email Spam Classifier")
st.write("Paste an email message below and find out if it's **Spam** or **Ham**.")

st.sidebar.header("Model Info")
st.sidebar.write(f"✅ Model Accuracy: **{acc*100:.2f}%**")
st.sidebar.write("Using Naive Bayes + Bag of Words")

# Input email text
email_text = st.text_area("✍️ Enter your email text here:", height=200)

if st.button("Predict"):
    if email_text.strip():
        text_vector = vectorizer.transform([email_text])
        prediction = model.predict(text_vector)[0]
        label = "🚨 Spam" if prediction == 1 else "✅ Ham"
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter some text before predicting.")




# ✅ How to Run

# Save this as app.py.

# Place emails.csv in the same folder.

# Install dependencies:

# pip install streamlit scikit-learn pandas


# Run the app:

# streamlit run app.py