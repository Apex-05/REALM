# nlp_pipeline/harmful_flag.py

from transformers import pipeline
import pandas as pd
import torch


DEVICE = 0 if torch.cuda.is_available() else -1


# ✅ CHANGED: Use a public, accessible model
HARM_PIPELINE = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",  # Public model, no authentication needed
    framework="pt",
    device=DEVICE,
    truncation=True,
    max_length=512
)


def minimal_clean_harmful(text: str) -> str:
    """Minimal cleaning for harmful content detection."""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    if len(text.split()) < 5:
        return ""
    
    return text


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


def aggregate_chunk_harmful(chunk_results: list) -> dict:
    """Aggregate harmful content predictions from multiple chunks."""
    if not chunk_results:
        return {'label': 'neutral', 'score': 0.0}
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    label_weights = {}
    for result in chunk_results:
        label = result['label']
        score = result['score']
        
        if label not in label_weights:
            label_weights[label] = 0.0
        label_weights[label] += score
    
    final_label = max(label_weights, key=label_weights.get)
    final_score = label_weights[final_label] / len(chunk_results)
    
    return {'label': final_label, 'score': final_score}


def run_harmful_content_flag(df: pd.DataFrame, batch_size: int = 32):
    """Run harmful content detection with chunking support."""
    
    # ✅ FIXED: Keep track of original comments
    original_comments = df['comment'].tolist()
    
    # Clean texts
    texts = df['comment'].fillna("").astype(str).apply(minimal_clean_harmful).tolist()
    
    # ✅ FIXED: Find valid indices (non-empty after cleaning)
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]
    
    if not valid_texts:
        return pd.DataFrame(columns=["comment", "harmful_label", "score"]), pd.Series(dtype=int)
    
    all_results = []
    
    for text in valid_texts:
        chunks = split_long_text(text, max_words=400)
        
        chunk_preds = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_preds = HARM_PIPELINE(batch, truncation=True, max_length=512)
                chunk_preds.extend(batch_preds)
            except Exception as e:
                print(f"Error processing batch: {e}")
                chunk_preds.extend([{'label': 'neutral', 'score': 0.5}] * len(batch))
        
        final_result = aggregate_chunk_harmful(chunk_preds)
        all_results.append(final_result)
    
    # ✅ FIXED: Build aligned lists
    result_comments = []
    result_labels = []
    result_scores = []
    
    for idx, result in zip(valid_indices, all_results):
        result_comments.append(original_comments[idx])
        result_labels.append(result['label'])
        result_scores.append(result['score'])
    
    # ✅ FIXED: Create DataFrame from aligned lists
    out_df = pd.DataFrame({
        "comment": result_comments,
        "harmful_label": result_labels,
        "score": result_scores
    })
    
    summary = out_df['harmful_label'].value_counts()
    return out_df, summary