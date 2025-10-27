# nlp_pipeline/sentiment.py

from transformers import pipeline
import pandas as pd
import torch
import re


DEVICE = 0 if torch.cuda.is_available() else -1


# ✅ Multilingual sentiment model with truncation enabled
SENT_PIPELINE = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    framework="pt",
    device=DEVICE,
    return_all_scores=True,  # ✅ THIS LINE IS CRITICAL
    truncation=True,
    max_length=512
)


def minimal_clean_sentiment(text: str) -> str:
    """
    Minimal cleaning for sentiment - your data is already well-cleaned!
    Just ensure it's a string and not empty.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Skip very short comments
    if len(text.split()) < 5:
        return ""
    
    # ✅ FIXED: Clean up newlines and extra whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    
    return text


def split_long_text(text: str, max_words: int = 400) -> list:
    """
    Split text into chunks if it's too long.
    DistilBERT max is 512 tokens, ~400 words is safe buffer.
    """
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def aggregate_chunk_sentiments(chunk_results: list) -> dict:
    """
    Aggregate sentiment predictions from multiple chunks.
    Uses weighted average based on confidence scores.
    """
    if not chunk_results:
        return {'label': 'neutral', 'score': 0.0}
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    # Count sentiment labels weighted by confidence
    sentiment_weights = {}
    
    for result in chunk_results:
        label = result['label']
        score = result['score']
        
        if label not in sentiment_weights:
            sentiment_weights[label] = 0.0
        sentiment_weights[label] += score
    
    # Get dominant sentiment
    final_label = max(sentiment_weights, key=sentiment_weights.get)
    final_score = sentiment_weights[final_label] / len(chunk_results)
    
    return {'label': final_label, 'score': final_score}


def run_sentiment_analysis(df: pd.DataFrame, batch_size: int = 32):
    """
    Run sentiment analysis on pre-cleaned data from filtered_batches.
    Handles long texts by chunking them.
    """
    # ✅ FIXED: Keep track of original comments
    original_comments = df['comment'].tolist()
    
    # Clean texts but keep indices aligned
    texts = df['comment'].fillna("").astype(str).apply(minimal_clean_sentiment).tolist()
    
    # ✅ FIXED: Find valid indices (non-empty after cleaning)
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]
    
    if not valid_texts:
        return pd.DataFrame(columns=["comment", "sentiment", "score"]), pd.Series(dtype=int)
    
    # Process texts and handle long sequences
    all_results = []
    
    for text in valid_texts:
        chunks = split_long_text(text, max_words=400)
        
        # Process chunks in batches
        chunk_preds = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_preds = SENT_PIPELINE(batch, truncation=True, max_length=512)

                # ✅ FIX: Select the top label from return_all_scores
                for pred_group in batch_preds:
                    if isinstance(pred_group, list):
                        best_pred = max(pred_group, key=lambda x: x['score'])
                        chunk_preds.append(best_pred)
                    else:
                        chunk_preds.append(pred_group)

            except Exception as e:
                print(f"Error processing batch: {e}")
                chunk_preds.extend([{'label': 'neutral', 'score': 0.5}] * len(batch))
        # Aggregate chunk results
        final_result = aggregate_chunk_sentiments(chunk_preds)
        all_results.append(final_result)
    
    # ✅ FIXED: Create lists with proper alignment
    result_comments = []
    result_sentiments = []
    result_scores = []
    
    # Only include rows that had valid text
    for idx, result in zip(valid_indices, all_results):
        result_comments.append(original_comments[idx])
        result_sentiments.append(result['label'])
        result_scores.append(result['score'])
    
    # ✅ FIXED: Create DataFrame from aligned lists
    out_df = pd.DataFrame({
        "comment": result_comments,
        "sentiment": result_sentiments,
        "score": result_scores
    })
    
    summary = out_df['sentiment'].value_counts()
    return out_df, summary