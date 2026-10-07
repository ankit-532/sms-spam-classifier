# src/predictor.py
# Model loading and prediction - EXACT same as Colab

import pickle
import os
from pathlib import Path
from .signals import get_structural_signals

# Global variables for loaded objects
_vectorizer = None
_model = None

def load_model(model_dir=None):
    """
    Load the saved TF-IDF vectorizer and Naive Bayes model.
    
    Args:
        model_dir (str): Path to the models directory.
                        If None, uses default path relative to this file.
    
    Returns:
        tuple: (vectorizer, model)
    """
    global _vectorizer, _model
    
    if model_dir is None:
        # Default: models directory at same level as src
        base_dir = Path(__file__).parent.parent
        model_dir = base_dir / "models"
    else:
        model_dir = Path(model_dir)
    
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    model_path = model_dir / "spam_model.pkl"
    
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(vectorizer_path, "rb") as f:
        _vectorizer = pickle.load(f)
    
    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    
    print(f"✅ Loaded vectorizer from: {vectorizer_path}")
    print(f"✅ Loaded model from: {model_path}")
    print(f"   Model classes: {_model.classes_}")
    print(f"   Vectorizer features: {len(_vectorizer.get_feature_names_out())}")
    
    return _vectorizer, _model

def predict_sms(message, verbose=False):
    """
    Predict if an SMS is SPAM, SUSPICIOUS, or HAM.
    
    Args:
        message (str): The SMS text to classify
        verbose (bool): If True, print detailed output
    
    Returns:
        dict: {
            "message": original message,
            "classification": "SPAM" | "SUSPICIOUS" | "HAM",
            "spam_probability": float (0.0 to 1.0),
            "signals": {
                "premium_number": bool,
                "shortcode": bool,
                "stop_language": bool
            },
            "decision_logic": str
        }
    """
    global _vectorizer, _model
    
    if _vectorizer is None or _model is None:
        load_model()
    
    # Step 1: Convert message to TF-IDF features
    message_tfidf = _vectorizer.transform([message])
    
    # Step 2: Get spam probability
    spam_prob = _model.predict_proba(message_tfidf)[0][1]
    
    # Step 3: Check structural signals
    signals = get_structural_signals(message)
    has_primary_signal = any(signals.values())
    
    # Step 4: Apply decision logic (0.20-0.50 zone)
    if spam_prob >= 0.50:
        classification = "SPAM"
        decision_logic = "high_confidence_spam"
    elif 0.20 <= spam_prob < 0.50:
        if has_primary_signal:
            classification = "SUSPICIOUS"
            decision_logic = "hybrid_escalated"
        else:
            classification = "HAM"
            decision_logic = "hybrid_not_escalated"
    else:  # spam_prob < 0.20
        classification = "HAM"
        decision_logic = "ham"
    
    result = {
        "message": message,
        "classification": classification,
        "spam_probability": round(spam_prob, 4),
        "signals": signals,
        "decision_logic": decision_logic
    }
    
    if verbose:
        print("=" * 60)
        print(f"📱 SMS: {message[:100]}{'...' if len(message) > 100 else ''}")
        print("-" * 60)
        print(f"📊 Spam Probability: {spam_prob*100:.1f}%")
        print(f"🔍 Signals:")
        print(f"   Premium Number: {'✅' if signals['premium_number'] else '❌'}")
        print(f"   Shortcode:      {'✅' if signals['shortcode'] else '❌'}")
        print(f"   Stop Language:  {'✅' if signals['stop_language'] else '❌'}")
        print(f"   Primary Signal: {'✅' if has_primary_signal else '❌'}")
        print("-" * 60)
        print(f"🏷️  Classification: {classification}")
        print(f"📌 Decision Logic: {decision_logic}")
        print("=" * 60)
    
    return result