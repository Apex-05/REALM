from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import torch

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# Extended stopword set for Reddit/social chat data
CUSTOM_STOPWORDS = {
    # Basic conversational and filler words
    'hi', 'hello', 'hey', 'gonna', 'wanna', 'got', 'get', 'went', 'going',
    'really', 'much', 'thing', 'things', 'stuff', 'lot', 'lots', 'way',
    'actually', 'probably', 'maybe', 'just', 'like', 'know', 'think',
    'want', 'need', 'make', 'said', 'say', 'people', 'time', 'day',

    # Pronouns and small words
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his',
    'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours',
    'they', 'them', 'their', 'theirs', 'a', 'an', 'the', 'and', 'but',
    'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'there', 'here',
    'what', 'where', 'when', 'who', 'whom', 'which', 'why', 'how',

    # Internet/chat slang and common short words
    'lol', 'omg', 'idk', 'wtf', 'u', 'ur', 'thx', 'k', 'ok', 'yeah',
    'yep', 'nah', 'yo', 'sup', 'pls', 'plz', 'btw', 'tbh', 'imo', 'fyi',
    'smh', 'brb', 'gtg', 'ikr', 'lmao', 'rofl', 'dm', 'bff', 'irl',
    'nsfw', 'edit', 'deleted', 'removed', 'mod', 'mods', 'reddit',
    'subreddit', 'op', 'xposted', 'crosspost', 'like', 'post', 'comment',
    'also', 'even', 'still', 'very', 'too', 'can', 'cannot', 'could',
    'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'did',
    'does', 'doing', 'done'
}
# Merge with sklearn English stop words
EXTENDED_STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

def minimal_clean_topic(text: str) -> str:
    """Preprocessing: skip non-string or very short texts (less than 5 words)."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text.split()) < 5:
        return ""
    return text

def extract_top_keywords(texts, n_keywords=10):
    """Extract global top keywords (stopwords/short words removed) using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(EXTENDED_STOPWORDS))
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    sums = np.array(X.sum(axis=0)).flatten()
    top_indices = sums.argsort()[::-1]
    filtered = []
    for i in top_indices:
        kw = features[i]
        # Remove any keyword in stopwords
        if kw in EXTENDED_STOPWORDS:
            continue
        # Remove phrases with any word < 3 chars
        if any(len(word) < 3 for word in kw.split()):
            continue
        filtered.append(kw)
        if len(filtered) >= n_keywords:
            break
    return filtered

def run_kmeans_topic_modeling(df: pd.DataFrame, n_clusters=10):
    texts = df["comment"].fillna("").astype(str).apply(minimal_clean_topic).tolist()
    valid_texts = [t for t in texts if t]
    if len(valid_texts) < n_clusters:
        return pd.DataFrame(columns=["comment", "topic", "topic_prob", "topic_label"]), pd.Series(dtype=int)

    embeddings = EMBED_MODEL.encode(valid_texts, convert_to_tensor=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    topic_labels = []
    for cluster_id in range(n_clusters):
        cluster_texts = [valid_texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        if cluster_texts:
            keywords = extract_top_keywords(cluster_texts, n_keywords=5)
            label = ", ".join(keywords)
        else:
            label = "No data"
        topic_labels.append(label)

    topic_labels_col = [topic_labels[label] if label >= 0 else "No dominant topic" for label in cluster_labels]
    distances = kmeans.transform(embeddings)
    min_distances = distances.min(axis=1)
    confidences = 1 / (1 + min_distances)

    out_df = pd.DataFrame({
        "comment": valid_texts,
        "topic": cluster_labels,
        "topic_prob": confidences,
        "topic_label": topic_labels_col
    })
    summary = out_df["topic"].value_counts()
    return out_df, summary

# Example usage:
# df = pd.read_csv("comments.csv")
# result_df, summary = run_kmeans_topic_modeling(df, n_clusters=10)
# result_df.to_csv("reddit_topics_labeled.csv", index=False)
