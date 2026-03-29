import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob

print("Loading comments...")
df = pd.read_csv("All_Comments_Final.csv")
comments = df['Comments'].fillna('').astype(str)

print(f"Total comments: {len(comments)}")

# Generate sentiment labels using TextBlob
print("Generating sentiment labels...")
def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 1  # Positive
        elif polarity < -0.1:
            return 0  # Negative
        else:
            return 2  # Neutral
    except:
        return 2  # Default to Neutral

sentiments = [get_sentiment(comment) for comment in comments]

print(f"Positive: {sentiments.count(1)}")
print(f"Negative: {sentiments.count(0)}")
print(f"Neutral: {sentiments.count(2)}")

# Train TF-IDF Vectorizer
print("\nTraining TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(comments)

# Train Naive Bayes classifier
print("Training Sentiment Model...")
model = MultinomialNB()
model.fit(X, sentiments)

# Save files
print("\nSaving model files...")
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✓ Model and vectorizer saved successfully!")
print("Ready to run the Streamlit app!")
