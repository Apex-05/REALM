# REALM: Real-Time Analysis of Linguistic Media

Realtime NLP Pipeline with Distributed PySpark Processing

This repository implements a **real-time NLP pipeline** designed for distributed processing using **Apache Spark** and modular NLP components.  
It performs text ingestion, preprocessing, and large-scale analysis on Reddit comments (or similar social data), supporting both **local** and **HDFS-based** distributed workflows.

***

## 1. Overview

The pipeline ingests and processes comment data from Reddit, performing a sequence of NLP tasks such as **sentiment analysis**, **emotion detection**, **harmful content flagging**, **keyword extraction**, **topic modeling**, and **named entity recognition (NER)**.

It supports two execution modes:

- **Local Mode (`nlp_pipeline/`)** – Runs all NLP modules as standard Python scripts. (These scripts take input data directly from HDFS storage, *not* local files.)
- **Distributed Mode (`nlp_pipeline_spark/`)** – Runs the same modules on **PySpark** for parallel and scalable execution. (Input data is also taken from HDFS storage.)

All ingestion batches are logged in a centralized log file, and outputs from both modes are stored in dedicated result directories.

***

## 2. Workflow Summary

1. **Data Streaming (Reddit or other sources)**  
   - Reddit comments are streamed and stored as text batch files.  
   - Raw and filtered text batches are maintained for structured data ingestion.  
   - Data flow is managed by `streaming/streaming.ipynb`.

2. **Storage and Logging**  
   - Each data batch is recorded in `logs/upload_logs.csv` with timestamps and metadata.  
   - Files are automatically uploaded to **HDFS** for distributed analysis.

3. **NLP Processing**  
   - `nlp_pipeline/` (local Python) and `nlp_pipeline_spark/` (PySpark) both consume batches from HDFS.  
   - Each module performs a specific NLP task, including sentiment, emotion, harmful content, topic modeling, keyword extraction, and entity extraction.

4. **Results Generation**  
   - Processed results are stored in `results/` (local) and `results_spark/` (distributed).  
   - Outputs are modular and aligned per batch for analysis and comparison.

5. **Visualization and Analytics**  
   - The `Analysis/` folder contains notebooks for result exploration and visualization.  
   - `analysis.ipynb` reads result CSVs to create visual summaries, while `topic_generator.ipynb` focuses on topic analysis.

***

## 3. Directory Structure

```
HDFS
|
|--- realtime_pipeline/raw_batches
|--- realtime_pipeline/filtered_batches

REALTIME-PIPELINE/
│
├── Analysis/
│   ├── analysis.ipynb              # Aggregates and visualizes NLP results
│   └── topic_generator.ipynb      # Generates and refines topic clusters
│
├── checkpoint/                    # Model checkpoints (ignored by Git)
├── checkpoints/
│
├── logs/
│   └── logs.csv                  # Batch upload and ingestion logs
│
├── nlp_pipeline/                 # Local (non-Spark) NLP modules
│   ├── __init__.py
│   ├── sentiment.py
│   ├── emotion.py
│   ├── harmful_flag.py
│   ├── topic_model.py
│   ├── keyword_extraction.py
│   └── entity_extraction.py
│
├── nlp_pipeline_spark/           # Spark-based distributed NLP jobs (.ipynb)
│   ├── sentiment_spark.ipynb
│   ├── emotion_spark.ipynb
│   ├── harmful_flag_spark.ipynb
│   ├── topic_spark.ipynb
│   ├── keywords.ipynb
│   └── entity.ipynb
│
├── reddit_streaming/             # Reddit comment ingestion
│   ├── raw_batches/              # Files uploaded to HDFS raw_batches, fed for topic modeling (.txt files)
│   ├── filtered_batches/         # Files uploaded to HDFS filtered_batches, fed for sentiment analysis (.txt files)
│
├── results/                      # Output CSVs from local NLP modules
├── results_spark/                # Output CSVs from distributed Spark modules
│
├── streaming/
│   └── streaming.ipynb           # Handles live Reddit data collection and batching
│
└── run_pipeline.py               # Unified entry point for local NLP execution
```
```
results/
|-- emotion_results.csv   	# columns: "comment","emotion","score"
|-- entities.csv          	# columns: "comment","entities"
|-- harmful_flags.csv     	# columns: "comment","harmful_label","score"
|-- keywords.csv          	# columns: "comment","keywords"
|-- sentiment_results.csv 	# columns: "comment","sentiment","score"
|-- topic_resluts.csv     	# columns: "comment","topic","topic_prob"


results_spark/
|-- emotion_flags_spark.csv  	# columns: "comment","emotion","score"
|-- entity_spark.csv         	# columns: "comment","entities"
|-- harmful_flags_spark.csv     # columns: "comment","harmful_label","score"
|-- keywords_spark.csv          # columns: "comment","keywords"
|-- sentiment_flags_spark.csv   # columns: "comment","sentiment","score"
|-- topic_results_spark.csv     # columns: "comment","topic","topic_prob"
```
***

## 4. Key Components

| File(s)                                    | Purpose / Function                                                                                                                                                           |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **streaming.ipynb**                        | Reddit streaming logic; cleans/comments, batches them, saves to HDFS, and logs batch metadata. Configurable cleaning, chunking, and batching strategies enable smooth ingestion. |
| **topic_generator.ipynb**                  | Notebook used for topic generation (takes input from `keywords_spark.csv` and `topic_results_spark.csv`)                                                                           |
| **analysis.ipynb**                         | Result exploration dashboard. Loads NLP CSV results, visualizes label distributions and entity occurrence, computes correlations, detects outliers, and aggregates dataset insights. |
| **run_pipeline.py**                        | Main pipeline entry point initializing all `.py files` in `npl_pipeline/`. Reads data from **HDFS**, triggers modular NLP tasks sequentially, aggregates and saves results for Sentiment, Emotion, Harmful content, Topic, Keywords, Entities. |
| **topic_model.py**                        | Generates semantic embeddings with SentenceTransformer (`all-MiniLM-L6-v2`), applies MiniBatchKMeans clustering, chooses optimal clusters using silhouette score, assigns topics per comment. |
| **sentiment.py**                          | Uses Hugging Face multilingual sentiment model. Processes batches, chunks long texts, aggregates predictions. Inputs taken directly from **HDFS** files.                      |
| **emotion.py**                            | Emotion detection using the `j-hartmann/emotion-english-distilroberta-base` transformer. Supports chunked batch inference and result aggregation.                              |
| **harmful_flag.py**                       | Harmful content flagging via RoBERTa (`SamLowe/roberta-base-goemotions`), chunked for long texts, provides score and classification. Inputs from HDFS.                            |
| **keyword_extraction.py**                 | Extracts keywords using RAKE and TF-IDF techniques. Filters common stopwords and noise, writes per-comment keywords and global themes.                                        |
| **entity_extraction.py**                  | Named entity extraction (NER) via spaCy (English Core Web Small model). Detects PERSON, ORG, LOC, etc., lists entities per comment and summary counts per type.                 |
| **sentiment_spark.ipynb**                  | PySpark distributed sentiment analysis using UDFs (uses Hugging Face multilingual sentiment model). Processes data in parallel from HDFS with optimized batching, outputs results as CSV in HDFS.                                |
| **emotion_spark.ipynb**                    | Spark distributed version of emotion detection. Implements UDFs for parallel inference, manages chunking and scoring efficiently on large datasets.                           |
| **harmful_flag_spark.ipynb**               | Spark UDF-based harmful content flagging using RoBERTa model. Processes HDFS datasets in parallel and outputs to CSV.                                                        |
| **topic_spark.ipynb**                      | Distributed topic modeling with optimized embedding generation in Spark, MiniBatchKMeans clustering, topic probability calculation, and HDFS CSV output.                      |
| **keywords.ipynb**                         | Interactive notebook for keyword extraction. Loads batches from HDFS/local fallback, extracts keywords with `RAKE/TF-IDF`, prints top keywords, and saves outputs.               |
| **entity.ipynb**                           | Interactive notebook for entity extraction from HDFS batches. Applies `spaCy NER model`, generates CSV outputs and entity type summaries.                                      |

***

## 5. Execution Flow

### A. Local Mode (for testing or small-to-medium datasets)  
```bash
python run_pipeline.py
```
- Runs all NLP modules sequentially on HDFS batch files.
- Suitable for local or small-scale executions without Spark cluster dependencies.

### B. Distributed Mode (for large-scale datasets)  
Run the corresponding Spark notebooks from `nlp_pipeline_spark/` using:  
```bash
spark-submit --master local[*] <notebook>.ipynb
```
- Supports distributed parallel processing on Spark clusters.
- Writes results into `results_spark/` directory.

***

## 6. Logging and Outputs

- **Logging:** Each batch ingestion and analysis step is logged in `logs/upload_logs.csv`.
- **Outputs:** Independent CSV files per NLP task are generated, such as:  
  - `sentiment_results.csv`  
  - `emotion_results.csv`  
  - `harmful_flags.csv`  
  - `topics.csv`  
  - `keywords.csv`  
  - `entities.csv`  
- Stored under `results/` for local mode, `results_spark/` for distributed mode.

***

## 7. Design Highlights

- **Dual-Mode Architecture:** Unified code base for local and distributed execution.
- **Modular Components:** Each NLP task is independently implementable and testable.
- **Scalable Processing:** Full HDFS integration and PySpark UDF support for large volumes.
- **Device Awareness:** Automatically uses GPU acceleration where available.
- **Unified Logging:** Comprehensive traceability for data batches and NLP results.
- **Extensible Design:** Easily adaptable for multilingual support and custom extensions.

***

## 8. Future Enhancements
- Implement Spark Streaming using Apache Kafka
- Real-time interactive dashboard for pipeline monitoring and analytics.
- Integration with streaming APIs for platforms like Twitter, YouTube.
- Advanced temporal topic tracking and trend visualization.
- Enhanced multilingual and domain-specific NLP model integration.
