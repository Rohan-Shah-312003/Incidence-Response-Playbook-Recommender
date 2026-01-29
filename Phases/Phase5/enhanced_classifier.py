# Enhanced Incident Classifier
# TF-IDF + Traditional ML, BERT, and Ensemble approaches

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import warnings

warnings.filterwarnings("ignore")


class EnhancedClassifier:
    """
    Multi-strategy classifier supporting:
    - Traditional ML (TF-IDF + Logistic Regression/SVM/NB)
    - BERT-based deep learning
    - Ensemble of multiple models
    """

    def __init__(self, strategy="ensemble"):
        """
        Args:
            strategy: 'tfidf', 'bert', or 'ensemble'
        """
        self.strategy = strategy
        self.vectorizer = None
        self.model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.label_map = {}
        self.reverse_label_map = {}

    def _create_tfidf_model(self):
        # Create TF-IDF based ensemble model
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),  # Capture up to trigrams
            min_df=2,
            max_df=0.85,
            max_features=5000,
            sublinear_tf=True,  # Use log scaling
            strip_accents="unicode",
        )

        # Ensemble of classifiers
        lr = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42, n_jobs=-1
        )

        svm = CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=42), cv=5
        )

        nb = MultinomialNB(alpha=0.1)

        # Weighted voting ensemble
        self.model = VotingClassifier(
            estimators=[("lr", lr), ("svm", svm), ("nb", nb)],
            voting="soft",
            weights=[2, 2, 1],  # Give more weight to LR and SVM
        )

    def _create_bert_model(self, num_labels):
        # Create BERT-based classifier
        model_name = "distilbert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )

    def _preprocess_for_bert(self, texts, labels=None):
        encodings = self.bert_tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        if labels is not None:
            dataset_dict = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": torch.tensor(labels),
            }
        else:
            dataset_dict = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }

        return Dataset.from_dict(dataset_dict)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (for BERT)
            y_val: Validation labels (for BERT)
        """
        # Create label mapping
        unique_labels = sorted(set(y_train))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}

        print(f"Training with strategy: {self.strategy}")
        print(f"Number of samples: {len(X_train)}")
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Classes: {unique_labels}\n")

        if self.strategy == "bert":
            self._train_bert(X_train, y_train, X_val, y_val)
        else:
            self._train_tfidf(X_train, y_train)

    def _train_tfidf(self, X_train, y_train):
        """Train TF-IDF based model"""
        print("Creating TF-IDF features...")
        self._create_tfidf_model()

        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"Feature matrix shape: {X_train_vec.shape}")

        print("Training ensemble model...")
        self.model.fit(X_train_vec, y_train)
        print("✓ Training complete!\n")

    def _train_bert(self, X_train, y_train, X_val, y_val):
        """Train BERT model"""
        print("Preparing BERT model...")

        # Convert labels to integers
        y_train_encoded = [self.label_map[label] for label in y_train]

        self._create_bert_model(len(self.label_map))

        # Prepare datasets
        print("Tokenizing texts...")
        train_dataset = self._preprocess_for_bert(X_train, y_train_encoded)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./models/bert_checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="no",
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
        )

        # Trainer
        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(self.bert_tokenizer),
        )

        print("Training BERT model...")
        trainer.train()
        print("✓ BERT training complete!\n")

    def predict(self, texts):
        """Predict class labels"""
        if self.strategy == "bert":
            return self._predict_bert(texts)
        else:
            return self._predict_tfidf(texts)

    def _predict_tfidf(self, texts):
        """Predict using TF-IDF model"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def _predict_bert(self, texts):
        """Predict using BERT model"""
        self.bert_model.eval()
        dataset = self._preprocess_for_bert(texts)

        predictions = []
        with torch.no_grad():
            for i in range(len(texts)):
                inputs = {
                    "input_ids": dataset[i]["input_ids"].unsqueeze(0),
                    "attention_mask": dataset[i]["attention_mask"].unsqueeze(0),
                }

                outputs = self.bert_model(**inputs)
                pred_label_idx = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(self.reverse_label_map[pred_label_idx])

        return predictions

    def predict_proba(self, texts):
        """Predict class probabilities"""
        if self.strategy == "bert":
            return self._predict_proba_bert(texts)
        else:
            return self._predict_proba_tfidf(texts)

    def _predict_proba_tfidf(self, texts):
        """Predict probabilities using TF-IDF model"""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def _predict_proba_bert(self, texts):
        """Predict probabilities using BERT model"""
        self.bert_model.eval()
        dataset = self._preprocess_for_bert(texts)

        all_probs = []
        with torch.no_grad():
            for i in range(len(texts)):
                inputs = {
                    "input_ids": dataset[i]["input_ids"].unsqueeze(0),
                    "attention_mask": dataset[i]["attention_mask"].unsqueeze(0),
                }

                outputs = self.bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)

        return np.array(all_probs)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        print("=" * 60)
        print(f"EVALUATION RESULTS ({self.strategy.upper()})")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confidence analysis
        max_probs = np.max(y_proba, axis=1)
        print(f"\nConfidence Analysis:")
        print(f"  Mean confidence: {np.mean(max_probs):.4f}")
        print(f"  Median confidence: {np.median(max_probs):.4f}")
        print(f"  Min confidence: {np.min(max_probs):.4f}")
        print(f"  Max confidence: {np.max(max_probs):.4f}")

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "predictions": y_pred,
            "probabilities": y_proba,
        }

    def save(self, path):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.strategy == "bert":
            # Save BERT model
            self.bert_model.save_pretrained(path / "bert_model")
            self.bert_tokenizer.save_pretrained(path / "bert_tokenizer")
            joblib.dump(
                {
                    "label_map": self.label_map,
                    "reverse_label_map": self.reverse_label_map,
                    "strategy": self.strategy,
                },
                path / "metadata.pkl",
            )
        else:
            # Save TF-IDF model
            joblib.dump(self.model, path / "classifier.pkl")
            joblib.dump(self.vectorizer, path / "vectorizer.pkl")
            joblib.dump(
                {
                    "label_map": self.label_map,
                    "reverse_label_map": self.reverse_label_map,
                    "strategy": self.strategy,
                },
                path / "metadata.pkl",
            )

        print(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model from disk"""
        path = Path(path)

        # Load metadata
        metadata = joblib.load(path / "metadata.pkl")
        strategy = metadata["strategy"]

        # Create instance
        instance = cls(strategy=strategy)
        instance.label_map = metadata["label_map"]
        instance.reverse_label_map = metadata["reverse_label_map"]

        if strategy == "bert":
            # Load BERT model
            instance.bert_model = AutoModelForSequenceClassification.from_pretrained(
                path / "bert_model"
            )
            instance.bert_tokenizer = AutoTokenizer.from_pretrained(
                path / "bert_tokenizer"
            )
        else:
            # Load TF-IDF model
            instance.model = joblib.load(path / "classifier.pkl")
            instance.vectorizer = joblib.load(path / "vectorizer.pkl")

        print(f"✓ Model loaded from {path}")
        return instance


def train_enhanced_model(
    data_path="./data/real_incidents_balanced.csv",
    strategy="ensemble",
    test_size=0.2,
    save_path="./models/enhanced",
):
    """
    Train enhanced classifier with specified strategy

    Args:
        data_path: Path to training data CSV
        strategy: 'tfidf', 'bert', or 'ensemble'
        test_size: Fraction of data for testing
        save_path: Where to save the trained model
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["incident_type"],
        test_size=test_size,
        random_state=42,
        stratify=df["incident_type"],
    )

    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}\n")

    # Train model
    classifier = EnhancedClassifier(strategy=strategy)
    classifier.fit(X_train, y_train)

    # Evaluate
    results = classifier.evaluate(X_test, y_test)

    # Save model
    classifier.save(save_path)

    return classifier, results


if __name__ == "__main__":
    # Example usage

    # Train TF-IDF ensemble model (fast, good performance)
    print("Training TF-IDF ensemble model...")
    classifier_tfidf, results_tfidf = train_enhanced_model(
        strategy="ensemble", save_path="./models/enhanced_tfidf"
    )

    # Uncomment to train BERT model (slower, potentially better)
    # print("\n\nTraining BERT model...")
    # classifier_bert, results_bert = train_enhanced_model(
    #     strategy='bert',
    #     save_path='./models/enhanced_bert'
    # )

    # Test prediction
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)

    test_incidents = [
        "Employee clicked on suspicious email link and entered credentials",
        "Unusual database queries detected from admin account after hours",
        "Multiple failed login attempts from external IP address",
    ]

    for incident in test_incidents:
        pred = classifier_tfidf.predict([incident])[0]
        proba = classifier_tfidf.predict_proba([incident])[0]
        confidence = max(proba)

        print(f"\nIncident: {incident[:60]}...")
        print(f"Prediction: {pred}")
        print(f"Confidence: {confidence:.2%}")
