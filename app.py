# app.py
import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load('logreg_best_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Depression Detection", page_icon="🧠", layout="centered")

st.title("🧠 Depression Detection from Reddit Text")
st.markdown("""
This simple app analyzes Reddit post text and predicts whether it shows signs of depression.  
*For demonstration purposes only — not a medical diagnosis tool.*
""")

# Text input
user_input = st.text_area("Enter Reddit post text:", height=200)

# Button
if st.button("Analyze"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text before analyzing.")
    else:
        # Preprocess input
        input_clean = clean_text(user_input)
        input_tfidf = vectorizer.transform([input_clean])
        prediction = model.predict(input_tfidf)[0]
        probability = model.predict_proba(input_tfidf)[0][1]  # probability of class 1

        # Display result
        if prediction == 1:
            st.error(f"⚠️ The post shows signs of depression. (Probability: {probability:.2f})")
        else:
            st.success(f"✅ The post does not show signs of depression. (Probability: {probability:.2f})")

# Footer
st.markdown("---")
st.caption("Developed as part of the Machine Learning Final Project. Not intended for clinical use.")