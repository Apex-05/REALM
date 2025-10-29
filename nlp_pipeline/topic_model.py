from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import torch
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words) if len(words) >= 5 else ""

def optimal_k(embeddings, max_k=8):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            n_init=5,
            random_state=42,
            batch_size=64
        )
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"[INFO] Optimal K ≈ {best_k} (silhouette={max(scores, key=lambda x: x[1])[1]:.3f})")
    return best_k

def run_kmeans_topic_modeling(df: pd.DataFrame, n_clusters=None):
    texts = df["comment"].fillna("").astype(str).apply(clean_text).tolist()
    valid_texts = [t for t in texts if t]
    if not valid_texts:
        return pd.DataFrame(columns=["comment", "topic", "topic_prob"]), pd.Series(dtype=int)

    print(f"[INFO] Encoding {len(valid_texts)} comments...")
    embeddings = EMBED_MODEL.encode(valid_texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings)  

    if n_clusters is None:
        n_clusters = min(optimal_k(embeddings, max_k=min(8, len(valid_texts)-1)), len(valid_texts))
    print(f"[INFO] Using {n_clusters} clusters")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=5,
        random_state=42,
        batch_size=64,
        max_iter=200,
        reassignment_ratio=0.05  
    )
    cluster_labels = kmeans.fit_predict(embeddings)

    centroids = normalize(kmeans.cluster_centers_)
    topic_probs = np.sum(embeddings * centroids[cluster_labels], axis=1)

    out_df = pd.DataFrame({
        "comment": valid_texts,
        "topic": cluster_labels,
        "topic_prob": topic_probs
    })
    summary = out_df["topic"].value_counts()

    return out_df, summary
