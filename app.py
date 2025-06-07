# Fake Job offer detector web app
import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    stopwords = set(["the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "with", "that", "of"])
    return ' '.join([word for word in text.split() if word not in stopwords])

# App UI
st.title("🕵️‍♂️ Fake Job Offer Detector")
st.write("Enter a job offer to check if it's a real or fake job posting:")

user_input = st.text_area("Job Offer", height=300)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job offer to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "✅ Genuine Offer" if prediction == 0 else "🚫 Fake Offer"
        st.success(label)
