# YouTube Sentiment Analyzer (NLP Final Project)

This project analyzes YouTube comments and predicts sentiment as **Positive**, **Negative**, or **Neutral** using a TF-IDF vectorizer and a machine learning classifier.

## Repository Requirements Checklist

This repository includes all required items:

- Python code files: [app.py](app.py), [train_model.py](train_model.py)
- Dataset files: [All_Comments_Final.csv](All_Comments_Final.csv), [Aggregated_Metrics_By_Video.csv](Aggregated_Metrics_By_Video.csv), [Aggregated_Metrics_By_Country_And_Subscriber_Status.csv](Aggregated_Metrics_By_Country_And_Subscriber_Status.csv), [Video_Performance_Over_Time.csv](Video_Performance_Over_Time.csv)
- Trained model files: [sentiment_model.pkl](sentiment_model.pkl), [tfidf_vectorizer.pkl](tfidf_vectorizer.pkl)
- GUI application code: [app.py](app.py) (Streamlit web app)
- Colab notebook: [Youtube_Channel_Analysis_NLP_Project_C_final.ipynb](Youtube_Channel_Analysis_NLP_Project_C_final.ipynb)

## Project Structure

- [app.py](app.py): Streamlit GUI for live sentiment prediction
- [train_model.py](train_model.py): Model training script and pickle export
- [sentiment_model.pkl](sentiment_model.pkl): Trained sentiment model
- [tfidf_vectorizer.pkl](tfidf_vectorizer.pkl): Trained TF-IDF vectorizer
- [All_Comments_Final.csv](All_Comments_Final.csv): Main comment dataset
- [Youtube_Channel_Analysis_NLP_Project_C_final.ipynb](Youtube_Channel_Analysis_NLP_Project_C_final.ipynb): Analysis notebook (Colab compatible)
- [Youtube_Channel_Analysis_NLP_Project_C_final.pdf](Youtube_Channel_Analysis_NLP_Project_C_final.pdf): Project report/exported notebook

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install streamlit scikit-learn pandas textblob
```

## Run the Web Application

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Retrain the Model (Optional)

If model files are missing or corrupted, retrain with:

```bash
python train_model.py
```

This regenerates:
- [sentiment_model.pkl](sentiment_model.pkl)
- [tfidf_vectorizer.pkl](tfidf_vectorizer.pkl)

## Colab Usage

You can open [Youtube_Channel_Analysis_NLP_Project_C_final.ipynb](Youtube_Channel_Analysis_NLP_Project_C_final.ipynb) directly in Google Colab by uploading it to Colab or selecting it from your GitHub repository once pushed.

## Notes

- The Streamlit app loads model files from the repository root.
- If pickle loading errors appear, run [train_model.py](train_model.py) to regenerate model artifacts.
