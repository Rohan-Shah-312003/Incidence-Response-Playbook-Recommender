import joblib

MODEL_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/tfidf.pkl"

classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def classify_incident(text: str):
    X = vectorizer.transform([text])
    probs = classifier.predict_proba(X)[0]

    idx = probs.argmax()
    label = classifier.classes_[idx]
    confidence = probs[idx]

    return label, float(confidence)
