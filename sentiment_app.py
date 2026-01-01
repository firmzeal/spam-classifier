import streamlit as st
from transformers import pipeline

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="😊 Sentiment Analysis AI", layout="wide")

# -----------------------
# Theme toggle
# -----------------------
theme = st.radio("Select Theme:", ("Light 🌞", "Dark 🌙"))
if theme == "Dark 🌙":
    st.markdown("""
        <style>
        .reportview-container {background-color: #0e1117; color: #ffffff;}
        .stButton>button {background-color: #1e2027; color: #ffffff;}
        .stTextArea>div>div>textarea {background-color: #1e2027; color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .reportview-container {background-color: #ffffff; color: #000000;}
        .stButton>button {background-color: #f0f2f6; color: #000000;}
        .stTextArea>div>div>textarea {background-color: #ffffff; color: #000000;}
        </style>
        """, unsafe_allow_html=True)

# -----------------------
# Page title
# -----------------------
st.title("😊 Sentiment Analysis AI")
st.markdown("Type a sentence below or click an example to see if it's **Positive or Negative**.")

# -----------------------
# Load Hugging Face sentiment model
# -----------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_model()

# -----------------------
# Example sentences
# -----------------------
examples = {
    "Positive": [
        "I love this movie", "This product is amazing", "I feel so happy today",
        "This is the best day ever", "Fantastic experience"
    ],
    "Negative": [
        "I hate this", "This is terrible", "I am very sad", 
        "Worst experience ever", "I am disappointed"
    ]
}

# -----------------------
# Initialize session state
# -----------------------
if 'user_text' not in st.session_state:
    st.session_state['user_text'] = ""

# -----------------------
# Layout: Columns
# -----------------------
col1, col2 = st.columns([3, 1])

with col1:
    user_text = st.text_area(
        "✍️ Type your sentence here:",
        value=st.session_state['user_text'],
        height=150,
        key="text_area"
    )

with col2:
    st.write("### Example Sentences")
    for label, texts in examples.items():
        st.write(f"#### {label} Examples")
        for msg in texts:
            if st.button(msg):
                st.session_state['user_text'] = msg

# -----------------------
# Reactive prediction
# -----------------------
if st.session_state['user_text'].strip() != "":
    results = sentiment_model(st.session_state['user_text'])[0]
    label = results['label']
    score = round(results['score']*100, 2)

    # Convert labels to simpler text
    if label == "POSITIVE":
        prediction = "Positive 😊"
    else:
        prediction = "Negative 😢"

    st.write(f"**Sentence:** {st.session_state['user_text']}")
    st.write(f"**Predicted Sentiment:** {prediction}")
    st.write(f"**Confidence:** {score}%")
