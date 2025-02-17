# Amazon-sentiment-Analysis
This Flask-based Sentiment Analysis App predicts whether a given review is Positive, Negative, or Neutral using three different machine learning models:
TF-IDF + Logistic Regression
Word2Vec + SVM
BERT + Logistic Regression

🚀 Features
✅ Accepts user input (review text) via a simple web UI
✅ Uses three different NLP models to predict sentiment
✅ Displays predictions for TF-IDF, Word2Vec, and BERT
✅ Supports Flask for Web Deployment
✅ Processes text using pre-trained models

⚙️ How It Works
1️⃣ User inputs a review via the UI
2️⃣ Preprocessing: Converts text to lowercase, removes stopwords & punctuation
3️⃣ Feature Extraction:
> TF-IDF Vectorization (for Logistic Regression)
> Word2Vec Embeddings (for SVM)
> BERT Embeddings (for Logistic Regression)
4️⃣ Prediction: The models output a sentiment score
5️⃣ Results are displayed in the UI

💡 Technologies Used
Flask (Backend Web Framework)
Scikit-Learn (Machine Learning)
Gensim (Word2Vec Embeddings)
Transformers (Hugging Face) (BERT Processing)
Pandas & NumPy (Data Processing)
HTML, CSS (Frontend UI)
