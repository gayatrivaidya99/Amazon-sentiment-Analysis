import pickle
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Load trained models and vectorizers
tfidf_model_path = "sentiment_model.pkl"
tfidf_vectorizer_path = "tfidf_vectorizer.pkl"
word2vec_model_path = "w2v_model.pkl"  # Actual Word2Vec model
word2vec_classifier_path = "w2v_svm.pkl"  # SVM trained on Word2Vec
bert_model_path = "bert_model.pkl"

# Load TF-IDF Model
with open(tfidf_model_path, "rb") as model_file:
    tfidf_model = pickle.load(model_file)

with open(tfidf_vectorizer_path, "rb") as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

# Load Word2Vec Model
word2vec_model = Word2Vec.load(word2vec_model_path)

# Load SVM Classifier trained on Word2Vec
with open(word2vec_classifier_path, "rb") as w2v_model_file:
    word2vec_classifier = pickle.load(w2v_model_file)

# Load BERT Model
with open(bert_model_path, "rb") as bert_model_file:
    bert_model_logreg = pickle.load(bert_model_file)

# Load BERT Tokenizer & Model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Sentiment Mapping
sentiment_labels = {-1: "Negative", +1: "Positive", 0: "Neutral"}

# Preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
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

# Load test dataset
test_data_path = "./dataset/test_data.csv"
df_test = pd.read_csv(test_data_path)

# Apply preprocessing to test data
df_test["cleaned_text"] = df_test["reviews.text"].apply(preprocess_text)

# Store predictions
predictions = []

for review in df_test["cleaned_text"]:
    review_entry = {"Review": review}

    # ======= Predict using TF-IDF Model =======
    processed_review_tfidf = tfidf_vectorizer.transform([review])
    prediction_tfidf = tfidf_model.predict(processed_review_tfidf)[0]
    review_entry["TF-IDF Prediction"] = sentiment_labels[prediction_tfidf]

    # ======= Predict using Word2Vec Model =======
    words = review.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if word_vectors:
        sentence_embedding = np.mean(word_vectors, axis=0).reshape(1, -1)
        prediction_w2v = word2vec_classifier.predict(sentence_embedding)[0]
        review_entry["Word2Vec Prediction"] = sentiment_labels[prediction_w2v]
    else:
        review_entry["Word2Vec Prediction"] = "Not Enough Words for Prediction"

    # ======= Predict using BERT Model =======
    processed_review_bert = get_bert_embedding(review).reshape(1, -1)
    prediction_bert = bert_model_logreg.predict(processed_review_bert)[0]
    review_entry["BERT Prediction"] = sentiment_labels[prediction_bert]

    predictions.append(review_entry)

# Convert to DataFrame and save
predictions_df = pd.DataFrame(predictions)
predictions_file_path = "predicted_sentiments.csv"
predictions_df.to_csv(predictions_file_path, index=False)

print(f"Predictions saved to: {predictions_file_path}")
