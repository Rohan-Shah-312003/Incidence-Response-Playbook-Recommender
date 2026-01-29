"""
Enhanced Similarity Recommender
Uses sentence embeddings + hybrid retrieval for better recommendations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import joblib
from typing import List, Tuple, Dict
import re


class EnhancedSimilarityRecommender:
    """
    Advanced similarity-based action recommender with:
    - Sentence embeddings (semantic similarity)
    - BM25 (keyword matching)
    - Time decay weighting
    - Multi-factor action scoring
    """

    def __init__(
        self,
        embedding_model="all-MiniLM-L6-v2",
        use_hybrid=True,
        time_decay_enabled=True,
    ):
        """
        Args:
            embedding_model: Sentence transformer model name
            use_hybrid: Combine dense + sparse retrieval
            time_decay_enabled: Weight recent incidents higher
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.use_hybrid = use_hybrid
        self.time_decay_enabled = time_decay_enabled

        self.df = None
        self.embeddings = None
        self.bm25 = None
        self.tokenized_corpus = None
        self.action_kb = None

        # Load action knowledge base
        kb_path = Path("knowledge/action_kb.json")
        if kb_path.exists():
            with open(kb_path) as f:
                self.action_kb = json.load(f)

    def fit(self, df: pd.DataFrame):
        """
        Index historical incidents

        Args:
            df: DataFrame with columns ['text', 'incident_type']
        """
        print("Indexing historical incidents...")
        self.df = df.copy()

        # Add timestamp if not present (for time decay)
        if "timestamp" not in self.df.columns:
            # Simulate timestamps (most recent = today, oldest = 1 year ago)
            n = len(self.df)
            base_date = datetime.now()
            self.df["timestamp"] = [
                base_date - timedelta(days=int(365 * i / n)) for i in range(n)
            ]

        # Generate sentence embeddings
        print("Generating sentence embeddings...")
        texts = self.df["text"].tolist()
        self.embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        # Prepare BM25 index if using hybrid retrieval
        if self.use_hybrid:
            print("Building BM25 index...")
            self.tokenized_corpus = [self._tokenize(text) for text in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✓ Indexed {len(self.df)} incidents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def _time_decay_weight(
        self,
        incident_date: datetime,
        current_date: datetime = None,
        half_life_days: int = 90,
    ) -> float:
        """
        Calculate time decay weight (exponential decay)

        Args:
            incident_date: When the incident occurred
            current_date: Reference date (default: now)
            half_life_days: Days for weight to decay to 50%
        """
        if current_date is None:
            current_date = datetime.now()

        days_diff = (current_date - incident_date).days
        return np.exp(-np.log(2) * days_diff / half_life_days)

    def _hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.7
    ) -> np.ndarray:
        """
        Hybrid search combining dense + sparse retrieval

        Args:
            query: Query text
            top_k: Number of results
            alpha: Weight for semantic similarity (1-alpha for BM25)

        Returns:
            Combined similarity scores
        """
        # Semantic similarity (dense)
        query_embedding = self.embedding_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]

        if not self.use_hybrid:
            return semantic_scores

        # BM25 scores (sparse)
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # Combine scores
        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

        return combined_scores

    def _score_action(
        self,
        action_id: str,
        action_freq: float,
        incident_similarity: float,
        classification_confidence: float,
        severity_multiplier: float = 1.0,
    ) -> float:
        """
        Multi-factor action scoring

        Args:
            action_id: Action identifier
            action_freq: Frequency in similar incidents
            incident_similarity: Similarity to query
            classification_confidence: Classification confidence
            severity_multiplier: Urgency modifier

        Returns:
            Combined action score
        """
        # Base weights
        weights = {
            "frequency": 0.3,
            "similarity": 0.3,
            "confidence": 0.2,
            "severity": 0.2,
        }

        score = (
            weights["frequency"] * action_freq
            + weights["similarity"] * incident_similarity
            + weights["confidence"] * classification_confidence
            + weights["severity"] * severity_multiplier
        )

        return score

    def recommend(
        self,
        incident_text: str,
        incident_type: str,
        classification_confidence: float,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Recommend actions for a new incident

        Args:
            incident_text: Description of the incident
            incident_type: Predicted incident type
            classification_confidence: Confidence of classification
            top_k: Number of similar incidents to retrieve
            alpha: Hybrid search weight

        Returns:
            (recommended_actions, similar_incidents)
        """
        # Get hybrid similarity scores
        similarity_scores = self._hybrid_search(incident_text, top_k, alpha)

        # Apply time decay if enabled
        if self.time_decay_enabled and "timestamp" in self.df.columns:
            current_date = datetime.now()
            time_weights = np.array(
                [
                    self._time_decay_weight(ts, current_date)
                    for ts in self.df["timestamp"]
                ]
            )
            similarity_scores = similarity_scores * time_weights

        # Get top K similar incidents
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        total_similarity = similarity_scores[top_indices].sum()

        # Collect similar incidents for evidence
        similar_incidents = []
        for idx in top_indices:
            similar_incidents.append(
                {
                    "incident_type": self.df.iloc[idx]["incident_type"],
                    "similarity": float(similarity_scores[idx]),
                    "text": self.df.iloc[idx]["text"][:500],
                }
            )

        # Score actions based on similar incidents
        action_scores = {}
        action_frequencies = {}

        # Map incident types to actions
        # ACTION_MAP = {
        #     "Phishing": ["IR-ID-01", "IR-CON-02", "IR-CON-03", "IR-POST-01"],
        #     "Insider Misuse": ["IR-ID-01", "IR-ID-02", "IR-CON-02", "IR-POST-01"],
        #     "Data Breach": ["IR-ID-01", "IR-CON-01", "IR-ERAD-01", "IR-POST-01"],
        #     "Malware": ["IR-ID-01", "IR-CON-01", "IR-ERAD-01", "IR-REC-01"],
        #     "Ransomware": [
        #         "IR-ID-01",
        #         "IR-CON-01",
        #         "IR-ERAD-01",
        #         "IR-REC-01",
        #         "IR-POST-01",
        #     ],
        # }
        ACTION_MAP = {
            "Phishing": [
                "IR-ID-01",
                "IR-CON-02",
                "IR-CON-03",
                "IR-ERAD-01",
                "IR-POST-01",
                "IR-POST-02",
            ],
            "Insider Misuse": [
                "IR-ID-01",
                "IR-ID-02",
                "IR-CON-02",
                "IR-ERAD-02",
                "IR-POST-01",
                "IR-POST-02",
            ],
            "Data Breach": [
                "IR-ID-01",
                "IR-ID-02",
                "IR-CON-01",
                "IR-ERAD-01",
                "IR-ERAD-02",
                "IR-POST-01",
            ],
            "Malware": [
                "IR-ID-01",
                "IR-ID-02",
                "IR-CON-01",
                "IR-ERAD-01",
                "IR-REC-01",
                "IR-POST-01",
            ],
            "Ransomware": [
                "IR-ID-01",
                "IR-CON-01",
                "IR-ERAD-01",
                "IR-REC-01",
                "IR-REC-02",
                "IR-POST-01",
            ],
            "Denial of Service": [
                "IR-ID-01",
                "IR-CON-01",
                "IR-REC-01",
                "IR-REC-02",
                "IR-POST-01",
            ],
        }

        for idx in top_indices:
            label = self.df.iloc[idx]["incident_type"]
            sim = similarity_scores[idx]

            for action_id in ACTION_MAP.get(label, []):
                if action_id not in self.action_kb:
                    continue

                action_frequencies[action_id] = action_frequencies.get(action_id, 0) + 1

                # Get severity multiplier based on phase
                phase = self.action_kb[action_id]["phase"]
                phase_weights = {
                    "Identification": 1.0,
                    "Containment": 0.9,
                    "Eradication": 0.7,
                    "Recovery": 0.6,
                    "Post-Incident": 0.4,
                }
                severity = phase_weights.get(phase, 0.5)

                # Calculate action score
                freq_norm = action_frequencies[action_id] / len(top_indices)
                score = self._score_action(
                    action_id, freq_norm, sim, classification_confidence, severity
                )

                action_scores[action_id] = action_scores.get(action_id, 0) + score

        # Normalize and rank actions
        max_score = max(action_scores.values()) if action_scores else 1.0

        recommended_actions = []
        for action_id, score in action_scores.items():
            if action_id not in self.action_kb:
                continue

            phase = self.action_kb[action_id]["phase"]

            # Phase ordering
            phase_order = {
                "Identification": 1,
                "Containment": 2,
                "Eradication": 3,
                "Recovery": 4,
                "Post-Incident": 5,
            }

            recommended_actions.append(
                {
                    "action_id": action_id,
                    "confidence": round((score / max_score) * 100, 2),
                    "phase": phase,
                    "phase_rank": phase_order.get(phase, 99),
                }
            )

        # Sort by phase, then by confidence
        recommended_actions.sort(key=lambda x: (x["phase_rank"], -x["confidence"]))

        return recommended_actions, similar_incidents

    def save(self, path: str):
        """Save recommender state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(path / "embeddings.npy", self.embeddings)

        # Save dataframe
        self.df.to_csv(path / "indexed_incidents.csv", index=False)

        # Save BM25 if used
        if self.use_hybrid:
            joblib.dump(self.bm25, path / "bm25.pkl")
            joblib.dump(self.tokenized_corpus, path / "tokenized_corpus.pkl")

        # Save metadata
        metadata = {
            "use_hybrid": self.use_hybrid,
            "time_decay_enabled": self.time_decay_enabled,
            "embedding_model_name": self.embedding_model.get_sentence_embedding_dimension(),
        }
        joblib.dump(metadata, path / "metadata.pkl")

        print(f"✓ Recommender saved to {path}")

    @classmethod
    def load(cls, path: str, embedding_model="all-MiniLM-L6-v2"):
        """Load recommender from disk"""
        path = Path(path)

        # Load metadata
        metadata = joblib.load(path / "metadata.pkl")

        # Create instance
        instance = cls(
            embedding_model=embedding_model,
            use_hybrid=metadata["use_hybrid"],
            time_decay_enabled=metadata["time_decay_enabled"],
        )

        # Load data
        instance.df = pd.read_csv(path / "indexed_incidents.csv")
        instance.df["timestamp"] = pd.to_datetime(instance.df["timestamp"])

        # Load embeddings
        instance.embeddings = np.load(path / "embeddings.npy")

        # Load BM25 if used
        if instance.use_hybrid:
            instance.bm25 = joblib.load(path / "bm25.pkl")
            instance.tokenized_corpus = joblib.load(path / "tokenized_corpus.pkl")

        print(f"✓ Recommender loaded from {path}")
        return instance


def build_recommender(
    data_path="./data/real_incidents_balanced.csv",
    save_path="./models/enhanced_similarity",
):
    """
    Build and save enhanced recommender
    """
    print("Building enhanced similarity recommender...")

    # Load data
    df = pd.read_csv(data_path)

    # Create recommender
    recommender = EnhancedSimilarityRecommender(
        embedding_model="all-MiniLM-L6-v2", use_hybrid=True, time_decay_enabled=True
    )

    # Index incidents
    recommender.fit(df)

    # Save
    recommender.save(save_path)

    return recommender


if __name__ == "__main__":
    # Build recommender
    recommender = build_recommender()

    # Test recommendations
    print("\n" + "=" * 60)
    print("TESTING RECOMMENDATIONS")
    print("=" * 60)

    test_incident = """
    During routine monitoring, repeated authentication attempts were observed 
    on a file server at 12:30 AM. The attempts originated from a valid internal 
    user account accessing sensitive directories.
    """

    actions, similar = recommender.recommend(
        incident_text=test_incident,
        incident_type="Insider Misuse",
        classification_confidence=0.85,
        top_k=5,
    )

    print("\nRecommended Actions:")
    for action in actions[:5]:
        print(
            f"  {action['action_id']} ({action['phase']}) - {action['confidence']:.1f}%"
        )

    print("\nSimilar Incidents:")
    for i, inc in enumerate(similar[:3], 1):
        print(f"\n{i}. [{inc['incident_type']}] (similarity: {inc['similarity']:.3f})")
        print(f"   {inc['text'][:100]}...")
