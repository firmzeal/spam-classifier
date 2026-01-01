import streamlit as st
import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------
# Function to clean text
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -----------------------
# Page title
# -----------------------
st.title("📧 Spam Message Classifier")
st.write("Type a message and the AI will tell you if it's Spam or Not Spam.")

# -----------------------
# Check if model already exists
# -----------------------
if os.path.exists("spam_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    # Load dataset
    data = pd.read_csv("spam.csv")
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data['message'] = data['message'].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Save model and vectorizer
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    # Show accuracy
    predictions = model.predict(X_test_vec)
    st.write("Model Accuracy:", round(accuracy_score(y_test, predictions)*100, 2), "%")

# -----------------------
# User input for prediction
# -----------------------
user_message = st.text_area("✍️ Type your message here:")

# -----------------------
# Predict button
# -----------------------
if st.button("Check Message"):
    if user_message.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_message = clean_text(user_message)
        message_vec = vectorizer.transform([clean_message])
        prediction = model.predict(message_vec)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT spam")
