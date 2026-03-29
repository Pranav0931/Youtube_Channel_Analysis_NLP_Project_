import streamlit as st
import pickle
import os

# -------------------------------
# Safe loading function
# -------------------------------
def load_file(filepath):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

# -------------------------------
# Load model & vectorizer
# -------------------------------
model = load_file("sentiment_model.pkl")
vectorizer = load_file("tfidf_vectorizer.pkl")

# Stop app if files not loaded
if model is None or vectorizer is None:
    st.stop()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="🎥")

# -------------------------------
# UI
# -------------------------------
st.title("🎥 YouTube Comment Sentiment Analyzer")
st.write("Analyze whether a comment is Positive, Negative, or Neutral")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("Built using TF-IDF + Machine Learning + Streamlit")

# Input
user_input = st.text_area("Enter a YouTube Comment:")

# Button
analyze = st.button("Analyze Sentiment")

# -------------------------------
# Prediction Logic
# -------------------------------
if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment first!")
    else:
        try:
            # Transform input
            transformed = vectorizer.transform([str(user_input)])

            # Predict
            prediction = model.predict(transformed)[0]

            # Debug (remove later if needed)
            st.write("Prediction value:", prediction)

            # -------------------------------
            # Handle different model outputs
            # -------------------------------
            if prediction in [1, "positive", "Positive"]:
                st.success("😊 Positive Sentiment")

            elif prediction in [0, "negative", "Negative"]:
                st.error("😡 Negative Sentiment")

            elif prediction in [2, "neutral", "Neutral"]:
                st.info("😐 Neutral Sentiment")

            else:
                st.warning(f"Unknown prediction output: {prediction}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
