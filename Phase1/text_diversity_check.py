import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df = pd.read_csv("data/incidents.csv")
texts = df["incident_description"].tolist()

print("[*] Loading SBERT model (simulating RAG embedding space)...")
# "all-MiniLM-L6-v2" is a standard, fast model for RAG use cases
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"[*] Encoding {len(texts)} descriptions...")
embeddings = model.encode(texts)

print("[*] Calculating cosine similarity matrix...")
similarity_matrix = cosine_similarity(embeddings)

# Mask diagonal (self-similarity is always 1.0)
np.fill_diagonal(similarity_matrix, 0)

# METRICS
max_sim = similarity_matrix.max()
mean_sim = similarity_matrix.mean()

# FIND THE OFFENDERS
# Let's find exactly WHICH two rows are too similar
rows, cols = np.where(similarity_matrix > 0.95)
duplicates = []
if len(rows) > 0:
    idx1, idx2 = rows[0], cols[0]
    duplicates = (texts[idx1], texts[idx2])

print(f"\n=== DIVERSITY REPORT ===")
print(f"Max Semantic Similarity: {max_sim:.4f}")
print(f"Avg Semantic Similarity: {mean_sim:.4f}")

# THRESHOLD LOGIC
# For SBERT, >0.95 usually means "Exact Semantic Duplicate"
# >0.85 means "Very similar topic/structure"
if max_sim > 0.95:
    print("\n[!] CRITICAL: Semantic collapse detected.")
    print(f"    Example collision ({max_sim:.4f}):")
    print(f"    1: {duplicates[0]}")
    print(f"    2: {duplicates[1]}")
elif max_sim > 0.90:
    print("\n[!] WARNING: High semantic similarity. Fine for classification, risky for RAG retrieval.")
else:
    print("\n[+] PASS: Dataset has good semantic separation.")