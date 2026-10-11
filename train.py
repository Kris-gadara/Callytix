import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

def train_model():
    print("Step 1: Downloading dataset from Kaggle...")
    try:
        # Load composite_train.csv from the Kaggle dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ibrahimbagwan12/composite-scam-transcript-dataset",
            "composite_train.csv"
        )
    except Exception as e:
        print(f"Error loading via load_dataset: {e}")
        print("Attempting alternate load using dataset_download...")
        path = kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset")
        train_file = os.path.join(path, "composite_train.csv")
        df = pd.read_csv(train_file)

    print(f"Dataset successfully loaded. Total rows: {len(df)}")
    
    # Check dataset columns
    print("Dataset columns:", df.columns.tolist())
    
    # Check and drop missing values
    df = df.dropna(subset=['text', 'label'])
    print(f"Rows after dropping NaNs: {len(df)}")

    print("\nStep 2: Cleaning text data (this may take a minute)...")
    # Clean the transcripts
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Keep only non-empty rows after cleaning
    df = df[df['cleaned_text'] != ""]
    print(f"Rows after text cleaning: {len(df)}")

    # Features and labels
    X = df['cleaned_text']
    y = df['label'].astype(int)

    print("\nStep 3: Performing Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nStep 4: Transforming text using TF-IDF...")
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nStep 5: Training Multinomial Naive Bayes classifier...")
    # Initialize and train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    print("\nStep 6: Evaluating the model on the test split...")
    y_pred = model.predict(X_test_tfidf)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 40)
    print("EVALUATION METRICS:")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 40)

    print("\nStep 7: Saving model and vectorizer...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print("Training pipeline complete!")
    
    # Return metrics for display in the app if called programmatically
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    train_model()
