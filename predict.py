import os
import joblib
import re
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# List of suspicious keywords for explanation/UI highlighting (not used for classification logic)
SUSPICIOUS_KEYWORDS = [
    "OTP", "Bank", "Account", "Verification", "KYC", "Refund", "UPI", 
    "PIN", "Card", "Customs", "Police", "Courier", "Lottery", "Investment", "Payment"
]

def load_prediction_assets():
    """
    Loads and returns the trained model and vectorizer.
    Raises FileNotFoundError if the models have not been trained yet.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Trained model or vectorizer not found. Please run train.py first to train the model."
        )
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_call(text: str):
    """
    Runs the complete prediction pipeline for a given text transcript.
    Returns a dictionary containing:
      - prediction: 'Fraudulent' or 'Legitimate'
      - confidence: float (percentage)
      - detected_keywords: list of matched suspicious keywords
    """
    # 1. Load saved model assets
    model, vectorizer = load_prediction_assets()
    
    # 2. Clean text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return {
            "prediction": "Legitimate",
            "confidence": 50.0,
            "detected_keywords": []
        }
    
    # 3. Transform text using vectorizer
    text_features = vectorizer.transform([cleaned_text])
    
    # 4. Predict class and get probability
    prediction_class = model.predict(text_features)[0]
    probabilities = model.predict_proba(text_features)[0]
    
    # Map class index to label name
    # 1 = Fraud/Scam, 0 = Legit
    label = "Fraudulent" if prediction_class == 1 else "Legitimate"
    
    # Confidence is the probability of the predicted class
    confidence = float(probabilities[prediction_class]) * 100.0
    
    # 5. Detect suspicious keywords in raw text (case-insensitive match)
    detected_keywords = []
    for kw in SUSPICIOUS_KEYWORDS:
        # Use regex to search for the keyword as a whole word boundary
        pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
        if pattern.search(text):
            detected_keywords.append(kw)
            
    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "detected_keywords": detected_keywords
    }
