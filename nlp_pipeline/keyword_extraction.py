import pandas as pd
import numpy as np
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize RAKE
RAKE = Rake(
    min_length=1,
    max_length=4,
    include_repeated_phrases=False
)

# Extended stopword set
CUSTOM_STOPWORDS = {
    'hi', 'hello', 'hey', 'gonna', 'wanna', 'got', 'get', 'went', 'going',
    'really', 'much', 'thing', 'things', 'stuff', 'lot', 'lots', 'way',
    'actually', 'probably', 'maybe', 'just', 'like', 'know', 'think',
    'want', 'need', 'make', 'said', 'say', 'people', 'time', 'day',
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his',
    'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours',
    'they', 'them', 'their', 'theirs', 'a', 'an', 'the', 'and', 'but',
    'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'there', 'here',
    'what', 'where', 'when', 'who', 'whom', 'which', 'why', 'how',
    'lol', 'omg', 'idk', 'wtf', 'u', 'ur', 'thx', 'k', 'ok', 'yeah',
    'yep', 'nah', 'yo', 'sup', 'pls', 'plz', 'btw', 'tbh', 'imo', 'fyi',
    'smh', 'brb', 'gtg', 'ikr', 'lmao', 'rofl', 'dm', 'bff', 'irl',
    'nsfw', 'edit', 'deleted', 'removed', 'mod', 'mods', 'reddit',
    'subreddit', 'op', 'xposted', 'crosspost', 'like', 'post', 'comment',
    'also', 'even', 'still', 'very', 'too', 'can', 'cannot', 'could',
    'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'did',
    'does', 'doing', 'done'
}

# Convert to list for TfidfVectorizer
STOPWORDS = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

def filter_topic_keywords(ranked_phrases_with_scores, min_score=1.0):
    """Filter RAKE keywords above score threshold and not in the stopword list."""
    filtered = []
    for item in ranked_phrases_with_scores:
        if isinstance(item, tuple) and len(item) == 2:
            score, phrase = item
            phrase_words = phrase.lower().split()
            # Remove any keyword containing stopwords
            if isinstance(score, float) and score >= min_score and all(word not in STOPWORDS for word in phrase_words):
                filtered.append(phrase)
        elif isinstance(item, str):  # If item is string, filter by stopwords
            phrase_words = item.lower().split()
            if all(word not in STOPWORDS for word in phrase_words):
                filtered.append(item)
    return filtered[:10]

def extract_tfidf_keywords(texts, n_keywords=5):
    """Extract top keywords using TF-IDF across all documents."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=STOPWORDS,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        avg_scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[::-1][:n_keywords]
        return [feature_names[i] for i in top_indices]
    except Exception as e:
        print(f"TF-IDF extraction error: {e}")
        return []

def run_keyword_extraction(df: pd.DataFrame, use_tfidf=True):
    """
    Extract topic-related keywords from pre-cleaned text.
    Only prints top 5 TF-IDF keywords to console, writes all per-row keywords to CSV.
    """
    original_comments = df['comment'].fillna("").astype(str).tolist()
    result_comments = []
    result_keywords = []
    valid_texts = []

    for text in original_comments:
        text = text.strip()
        if not text or len(text.split()) < 5:
            continue
        valid_texts.append(text)

        try:
            RAKE.extract_keywords_from_text(text)
            ranked_phrases = RAKE.get_ranked_phrases_with_scores()
            topic_keywords = filter_topic_keywords(ranked_phrases, min_score=1.0)
            keywords_str = ", ".join(topic_keywords) if topic_keywords else "No keywords found"
            result_comments.append(text)
            result_keywords.append(keywords_str)
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            result_comments.append(text)
            result_keywords.append("Error processing")


    # Output all results to DataFrame
    out_df = pd.DataFrame({
        "comment": result_comments,
        "keywords": result_keywords
    })

    # Create summary - count all keywords per comment for extra analysis (not printed)
    all_keywords = []
    for kw_str in result_keywords:
        if kw_str and kw_str not in ["No keywords found", "Error processing"]:
            all_keywords.extend([kw.strip() for kw in kw_str.split(", ")])
    summary = pd.Series(all_keywords).value_counts() if all_keywords else pd.Series(dtype=int)

    return out_df, summary

def extract_topic_themes(df: pd.DataFrame, n_topics=5):
    """Print and return the most frequent keywords (topic themes) from all comments."""
    all_keywords = []
    for kw_str in df['keywords']:
        if kw_str and kw_str not in ["No keywords found", "Error processing"]:
            all_keywords.extend([kw.strip() for kw in kw_str.split(", ")])
    keyword_counts = Counter(all_keywords)
    top_themes = keyword_counts.most_common(n_topics)
    print(f"\n🔍 Top {n_topics} Topic Themes:")
    for theme, count in top_themes:
        print(f"  - {theme}: {count} occurrences")
    return top_themes


