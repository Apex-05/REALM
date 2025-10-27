# nlp_pipeline/emotion.py

from transformers import pipeline
import pandas as pd
import torch


DEVICE = 0 if torch.cuda.is_available() else -1


EMOTION_PIPELINE = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    framework="pt",
    device=DEVICE,
    truncation=True,
    max_length=512
)


def split_long_text(text: str, max_words: int = 400) -> list:
    """Split text into chunks if it's too long."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def aggregate_chunk_emotions(chunk_results: list) -> dict:
    """Aggregate by finding the most common emotion."""
    if not chunk_results:
        return {'label': 'neutral', 'score': 0.0}
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    emotion_weights = {}
    for result in chunk_results:
        label = result['label']
        score = result['score']
        if label not in emotion_weights:
            emotion_weights[label] = 0.0
        emotion_weights[label] += score
    
    final_label = max(emotion_weights, key=emotion_weights.get)
    final_score = emotion_weights[final_label] / len(chunk_results)
    
    return {'label': final_label, 'score': final_score}


def run_emotion_detection(df: pd.DataFrame, batch_size: int = 32):
    """Run emotion detection with chunking support."""
    
    # ✅ FIXED: Keep track of original comments
    original_comments = df['comment'].tolist()
    
    # Clean texts
    texts = df['comment'].fillna("").astype(str).tolist()
    texts = [t.strip() for t in texts if t.strip() and len(t.split()) >= 5]
    
    if not texts:
        return pd.DataFrame(columns=["comment", "emotion", "score"]), pd.Series(dtype=int)
    
    all_results = []
    
    for text in texts:
        chunks = split_long_text(text, max_words=400)
        chunk_preds = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_preds = EMOTION_PIPELINE(batch, truncation=True, max_length=512)
                chunk_preds.extend(batch_preds)
            except Exception as e:
                print(f"Emotion detection error: {e}")
                chunk_preds.extend([{'label': 'neutral', 'score': 0.5}] * len(batch))
        
        final_result = aggregate_chunk_emotions(chunk_preds)
        all_results.append(final_result)
    
    # ✅ FIXED: Build aligned lists with results
    result_comments = []
    result_emotions = []
    result_scores = []
    
    for text, result in zip(texts, all_results):
        result_comments.append(text)
        result_emotions.append(result['label'])
        result_scores.append(result['score'])
    
    # ✅ FIXED: Create DataFrame from aligned lists
    out_df = pd.DataFrame({
        "comment": result_comments,
        "emotion": result_emotions,
        "score": result_scores
    })
    
    summary = out_df['emotion'].value_counts()
    return out_df, summary