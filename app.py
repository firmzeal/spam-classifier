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
# Page layout
# -----------------------
st.set_page_config(page_title="📧 Spam Classifier", layout="wide")

# -----------------------
# Theme Toggle
# -----------------------
theme = st.radio("Select Theme:", ("Light 🌞", "Dark 🌙"))

if theme == "Dark 🌙":
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #1e2027;
            color: #ffffff;
        }
        .stTextArea>div>div>textarea {
            background-color: #1e2027;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton>button {
            background-color: #f0f2f6;
            color: #000000;
        }
        .stTextArea>div>div>textarea {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)

# -----------------------
# Page title & description
# -----------------------
st.title("📧 Spam Message Classifier")
st.markdown("""
Type a message or select an example below. The AI will predict instantly if it's **Spam** or **Not Spam**.
""")

# -----------------------
# Load or train model
# -----------------------
if os.path.exists("spam_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    data = pd.read_csv("spam.csv")
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data['message'] = data['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    predictions = model.predict(X_test_vec)
    st.info(f"Model Accuracy: {round(accuracy_score(y_test, predictions)*100, 2)}%")

# -----------------------
# Layout: Columns
# -----------------------
col1, col2 = st.columns([3, 1])

with col1:
    user_message = st.text_area("✍️ Type your message here:", height=150)

with col2:
    st.write("### Example Messages")
    examples = [
        "Win a free iPhone now!!!",
        "Are you coming to class today?",
        "Claim your free prize",
        "Don't forget our meeting tomorrow",
        "URGENT! You won cash reward"
    ]
    for msg in examples:
        if st.button(msg):
            st.session_state['user_message'] = msg

# -----------------------
# Reactive message handling
# -----------------------
if 'user_message' not in st.session_state:
    st.session_state['user_message'] = user_message
else:
    if user_message != st.session_state['user_message']:
        st.session_state['user_message'] = user_message

# -----------------------
# Auto-predict instantly
# -----------------------
if st.session_state['user_message'].strip() != "":
    clean_message = clean_text(st.session_state['user_message'])
    message_vec = vectorizer.transform([clean_message])
    prediction = model.predict(message_vec)
    probability = model.predict_proba(message_vec)[0]

    if prediction[0] == 1:
        st.error(f"🚨 SPAM! Probability: {round(probability[1]*100, 2)}%")
    else:
        st.success(f"✅ Not Spam. Probability: {round(probability[0]*100, 2)}%")
