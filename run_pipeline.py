import os
from pathlib import Path
import pandas as pd
import subprocess
import warnings
warnings.filterwarnings('ignore')

from nlp_pipeline import (
    sentiment,
    emotion,
    topic_model,
    keyword_extraction,
    entity_extraction,
    harmful_flag
)

BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HDFS_CMD = r"E:\hadoop\bin\hdfs.cmd"
hdfs_uri = "hdfs://localhost:9000"
HDFS_SENTIMENT_PATH = f"{hdfs_uri}/user/adarsh/realtime_pipeline/filtered_batches"
HDFS_TOPIC_PATH = f"{hdfs_uri}/user/adarsh/realtime_pipeline/raw_batches"

def read_hdfs_all_text(hdfs_path: str, hdfs_cmd: str, split_by_newline: bool = True):
    """Read all .txt files from HDFS and split by newlines for individual comments."""
    try:
        list_cmd = [hdfs_cmd, "dfs", "-ls", hdfs_path]
        list_out = subprocess.check_output(list_cmd, stderr=subprocess.DEVNULL)
        list_text = list_out.decode("utf-8", errors="replace")

        file_paths = []
        for line in list_text.split("\n"):
            if ".txt" in line:
                parts = line.split()
                if parts:
                    file_path = parts[-1]
                    if file_path.startswith(hdfs_uri):
                        file_paths.append(file_path)

        documents = []
        for file_path in file_paths:
            try:
                cmd = [hdfs_cmd, "dfs", "-cat", file_path]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                text = out.decode("utf-8", errors="replace").strip()
                if not text or len(text.split()) < 5:
                    continue
                if split_by_newline:
                    comments = [c.strip() for c in text.split('\n') if c.strip() and len(c.split()) >= 5]
                    documents.extend(comments)
                else:
                    documents.append(text)
            except Exception:
                continue
        return documents
    except Exception:
        return []

def summary_line(label, filename, dataframe, main_col=None, extra=None):
    if dataframe is None or len(dataframe) == 0:
        return f"{label.upper()} ({filename}): No data."
    line = f"{label.upper()} ({filename}): Rows: {len(dataframe)}"
    if main_col and main_col in dataframe.columns:
        vc = dataframe[main_col].value_counts().to_dict()
        flat = ", ".join([f"{k}: {v}" for k, v in vc.items()])
        line += " | " + flat
    if extra:
        line += " | " + extra
    return line

def main():
    # SENTIMENT PIPELINE
    sentiment_lines = read_hdfs_all_text(HDFS_SENTIMENT_PATH, HDFS_CMD, split_by_newline=True)
    sent_df = pd.DataFrame({"comment": sentiment_lines}) if sentiment_lines else None
    sent_result, sent_summary = (sentiment.run_sentiment_analysis(sent_df) if sent_df is not None and not sent_df.empty else (None, None))
    if sent_result is not None and not sent_result.empty:
        sent_result.to_csv(RESULTS_DIR / "sentiment_results.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(summary_line("sentiment analysis", "sentiment_results.csv", sent_result, "sentiment"))

    emo_df, emo_summary = (emotion.run_emotion_detection(sent_df) if sent_df is not None and not sent_df.empty else (None, None))
    if emo_df is not None and not emo_df.empty:
        emo_df.to_csv(RESULTS_DIR / "emotion_results.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(summary_line("emotion analysis", "emotion_results.csv", emo_df, "emotion"))

    harm_df, harm_summary = (harmful_flag.run_harmful_content_flag(sent_df) if sent_df is not None and not sent_df.empty else (None, None))
    if harm_df is not None and not harm_df.empty:
        harm_df.to_csv(RESULTS_DIR / "harmful_flags.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(summary_line("harmful content", "harmful_flags.csv", harm_df, "harmful_label"))

    # TOPIC PIPELINE
    topic_lines = read_hdfs_all_text(HDFS_TOPIC_PATH, HDFS_CMD, split_by_newline=True)
    topic_df = pd.DataFrame({"comment": topic_lines}) if topic_lines else None

    topic_result, topic_summary = (topic_model.run_kmeans_topic_modeling(topic_df) if topic_df is not None and not topic_df.empty else (None, None))
    if topic_result is not None and not topic_result.empty:
        topic_result.to_csv(RESULTS_DIR / "topic_results.csv", index=False, encoding="utf-8-sig", quoting=1)
    print(summary_line("topic modeling", "topic_results.csv", topic_result, "topic"))

    kw_df, kw_summary = (keyword_extraction.run_keyword_extraction(topic_df) if topic_df is not None and not topic_df.empty else (None, None))
    if kw_df is not None and not kw_df.empty:
        kw_df.to_csv(RESULTS_DIR / "keywords.csv", index=False, encoding="utf-8-sig", quoting=1)
    keyword_text = f"Unique keywords: {len(kw_summary)}" if kw_summary is not None else ""
    print(summary_line("keywords", "keywords.csv", kw_df, extra=keyword_text))

    ent_df, ent_summary = (entity_extraction.run_entity_extraction(topic_df) if topic_df is not None and not topic_df.empty else (None, None))
    if ent_df is not None and not ent_df.empty:
        ent_df.to_csv(RESULTS_DIR / "entities.csv", index=False, encoding="utf-8-sig", quoting=1)
    ent_cols = []
    if ent_summary is not None:
        ent_cols = [f"{k}: {v}" for k, v in ent_summary.to_dict().items()]
    print(summary_line("entity extraction", "entities.csv", ent_df, extra=", ".join(ent_cols)))

if __name__ == "__main__":
    main()
