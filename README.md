# Amazon-sentiment-Analysis
This Flask-based Sentiment Analysis App predicts whether a given review is Positive, Negative, or Neutral using three different machine learning models:
  1. TF-IDF + Logistic Regression
  2. Word2Vec + SVM
  3. BERT + Logistic Regression

**Features**
  1. Accepts user input (review text) via a simple web UI
  2. Uses three different NLP models to predict sentiment
  3. Displays predictions for TF-IDF, Word2Vec, and BERT
  4. Supports Flask for Web Deployment
  5. Processes text using pre-trained models

**How It Works**
  1. User inputs a review via the UI
  2. Preprocessing: Converts text to lowercase, removes stopwords & punctuation
  3. Feature Extraction:
    > TF-IDF Vectorization (for Logistic Regression)
    > Word2Vec Embeddings (for SVM)
    > BERT Embeddings (for Logistic Regression)
  4. Prediction: The models output a sentiment score
  5. Results are displayed in the UI

**Technologies Used**
  1. Flask (Backend Web Framework)
  2. Scikit-Learn (Machine Learning)
  3. Gensim (Word2Vec Embeddings)
  4. Transformers (Hugging Face) (BERT Processing)
  5. Pandas & NumPy (Data Processing)
  6. HTML, CSS (Frontend UI)
