"""
Enhanced classifier using the new model architecture
"""

import joblib
from pathlib import Path

# Try to load enhanced model first, fall back to old model
MODEL_PATH = Path("models/enhanced_tfidf")
OLD_MODEL_PATH = "models/classifier.pkl"
OLD_VECTORIZER_PATH = "models/tfidf.pkl"

# Check if enhanced model exists
if MODEL_PATH.exists():
    print("Loading enhanced ensemble model...")
    from Phases.Phase5.enhanced_classifier import EnhancedClassifier

    classifier = EnhancedClassifier.load(MODEL_PATH)
    _use_enhanced = True
else:
    print("Enhanced model not found, using legacy model...")
    classifier = joblib.load(OLD_MODEL_PATH)
    vectorizer = joblib.load(OLD_VECTORIZER_PATH)
    _use_enhanced = False


def classify_incident(text: str):
    """
    Classify incident and return (label, confidence)

    Args:
        text: Incident description

    Returns:
        (predicted_label, confidence_score)
    """
    if _use_enhanced:
        # Use enhanced model
        label = classifier.predict([text])[0]
        probs = classifier.predict_proba([text])[0]

        # Get confidence for predicted label
        label_idx = list(classifier.label_map.keys()).index(label)
        confidence = probs[label_idx]
    else:
        # Use legacy model
        X = vectorizer.transform([text])
        probs = classifier.predict_proba(X)[0]

        idx = probs.argmax()
        label = classifier.classes_[idx]
        confidence = probs[idx]

    return label, float(confidence)
