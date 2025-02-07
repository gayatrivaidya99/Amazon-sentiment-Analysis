from flask import Flask, render_template, request
import numpy as np
import pickle
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import os

# Flask App Initialization
app = Flask(__name__, template_folder="template")

# Sentiment Mapping
sentiment_labels = {-1: "Negative", +1: "Positive", 0: "Neutral"}

# File Paths for Models
MODEL_DIR = "./"
tfidf_model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
word2vec_model_path = os.path.join(MODEL_DIR, "w2v_model.pkl")
word2vec_classifier_path = os.path.join(MODEL_DIR, "w2v_svm.pkl")
bert_model_path = os.path.join(MODEL_DIR, "bert_model.pkl")

# Load Models
def load_model(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    else:
        print(f"⚠️ Warning: Model file '{filepath}' not found.")
        return None

# Load models
tfidf_model = load_model(tfidf_model_path)
tfidf_vectorizer = load_model(tfidf_vectorizer_path)
word2vec_model = Word2Vec.load(word2vec_model_path) if os.path.exists(word2vec_model_path) else None
word2vec_classifier = load_model(word2vec_classifier_path)
bert_model_logreg = load_model(bert_model_path)

# Load BERT Tokenizer & Model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    tokens = text.split()
    stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "this", "that", "to", "of"}
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = bert_tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token embedding

@app.route("/", methods=["GET", "POST"])
def index():
    tfidf_sentiment = None
    word2vec_sentiment = None
    bert_sentiment = None
    error = None

    if request.method == "POST":
        user_review = request.form["review"].strip()

        if not user_review:
            error = "Please enter a valid review."
            return render_template("index.html", error=error)

        cleaned_review = preprocess_text(user_review)

        # ======= Predict using TF-IDF Model =======
        if tfidf_model and tfidf_vectorizer:
            processed_review_tfidf = tfidf_vectorizer.transform([cleaned_review])
            prediction_tfidf = tfidf_model.predict(processed_review_tfidf)[0]
            tfidf_sentiment = sentiment_labels.get(prediction_tfidf, "Unknown")
        else:
            tfidf_sentiment = "Model Not Loaded"

        # ======= Predict using Word2Vec Model =======
        if word2vec_model and word2vec_classifier:
            words = cleaned_review.split()
            word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0).reshape(1, -1)
                prediction_w2v = word2vec_classifier.predict(sentence_embedding)[0]
                word2vec_sentiment = sentiment_labels.get(prediction_w2v, "Unknown")
            else:
                word2vec_sentiment = "Not Enough Words for Prediction"
        else:
            word2vec_sentiment = "Model Not Loaded"

        # ======= Predict using BERT Model =======
        if bert_model_logreg:
            processed_review_bert = get_bert_embedding(cleaned_review).reshape(1, -1)
            prediction_bert = bert_model_logreg.predict(processed_review_bert)[0]
            bert_sentiment = sentiment_labels.get(prediction_bert, "Unknown")
        else:
            bert_sentiment = "Model Not Loaded"

        # Debugging logs
        print(f"\n🔍 User Input: {user_review}")
        print(f"📌 TF-IDF Prediction: {tfidf_sentiment}")
        print(f"📌 Word2Vec Prediction: {word2vec_sentiment}")
        print(f"📌 BERT Prediction: {bert_sentiment}\n")

    return render_template(
        "index.html",
        tfidf_sentiment=tfidf_sentiment,
        word2vec_sentiment=word2vec_sentiment,
        bert_sentiment=bert_sentiment,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
