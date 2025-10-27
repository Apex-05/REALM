# nlp_pipeline/entity_extraction.py

import pandas as pd
import spacy


# Load small English model
NLP = spacy.load("en_core_web_sm")


def run_entity_extraction(df: pd.DataFrame):
    """Extract named entities from pre-cleaned text."""
    
    # ✅ FIXED: Keep track of original comments
    original_comments = df['comment'].tolist()
    
    # Process each comment
    result_comments = []
    result_entities = []
    
    for text in df['comment'].fillna("").astype(str):
        text = text.strip()
        
        # Skip empty texts
        if not text or len(text.split()) < 5:
            continue
        
        try:
            doc = NLP(text)
            ent_list = [f"{ent.text}:{ent.label_}" for ent in doc.ents]
            entities_str = ", ".join(ent_list) if ent_list else "No entities found"
            
            result_comments.append(text)
            result_entities.append(entities_str)
            
        except Exception as e:
            print(f"Error processing text: {e}")
            result_comments.append(text)
            result_entities.append("Error processing")
    
    # ✅ FIXED: Create DataFrame from aligned lists
    out_df = pd.DataFrame({
        "comment": result_comments,
        "entities": result_entities
    })
    
    # Create summary - count entity types
    entity_types = {}
    for ent_str in result_entities:
        if ent_str and ent_str != "No entities found" and ent_str != "Error processing":
            for entity_pair in ent_str.split(", "):
                if ":" in entity_pair:
                    ent_text, ent_label = entity_pair.rsplit(":", 1)
                    if ent_label not in entity_types:
                        entity_types[ent_label] = 0
                    entity_types[ent_label] += 1
    
    summary = pd.Series(entity_types) if entity_types else pd.Series(dtype=int)
    
    return out_df, summary