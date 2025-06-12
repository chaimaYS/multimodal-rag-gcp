# Multimodal RAG System — CLIP + Gemini + BigQuery

A production-grade **Multimodal Retrieval-Augmented Generation** pipeline on Google Cloud Platform. Uses CLIP for image-text embeddings, BigQuery for vector storage and retrieval, and Gemini for intelligent image summarization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│                                                                     │
│   Images ──▶ CLIP Encoder ──▶ Image Embeddings ──┐                 │
│                                                    │                │
│   Text   ──▶ CLIP Encoder ──▶ Text Embeddings  ──┤                 │
│                                                    ▼                │
│                                            ┌──────────────┐        │
│   Images ──▶ Gemini Vision ──▶ Summaries ──▶│   BigQuery   │        │
│                                             │  Vector Store │        │
│                                             └──────┬───────┘        │
└────────────────────────────────────────────────────┼────────────────┘
                                                     │
┌────────────────────────────────────────────────────┼────────────────┐
│                        RETRIEVAL LAYER             │                │
│                                                    ▼                │
│   User Query ──▶ CLIP Encode ──▶ Vector Search (cosine similarity) │
│                                         │                           │
│                                         ▼                           │
│                                  Top-K Results                      │
│                                  (images + summaries)               │
│                                         │                           │
│                                         ▼                           │
│                                  Gemini LLM ──▶ Final Answer       │
│                                  (RAG response with context)        │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | OpenAI CLIP (ViT-B/32) |
| Image Summarization | Google Gemini 1.5 Pro (Vision) |
| Vector Storage | BigQuery (VECTOR_SEARCH) |
| Object Storage | Google Cloud Storage (GCS) |
| Orchestration | Apache Airflow (Cloud Composer) |
| Language | Python 3.10+ |
| Infrastructure | Terraform (GCP) |

## Project Structure

```
multimodal-rag-gcp/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── clip_encoder.py          # CLIP embedding generation
│   ├── gemini_summarizer.py     # Gemini Vision image summarization
│   ├── bigquery_vector_store.py # BigQuery storage & vector search
│   ├── rag_pipeline.py          # End-to-end RAG retrieval
│   └── ingestion_pipeline.py    # Batch ingestion orchestration
├── config/
│   └── config.yml               # Project configuration
├── notebooks/
│   └── demo_rag_query.ipynb     # Interactive demo notebook
├── tests/
│   └── test_pipeline.py         # Unit tests
└── docs/
    └── design_decisions.md      # Architecture decisions
```

## Quick Start

```bash
# Clone
git clone https://github.com/chaimaYS/multimodal-rag-gcp.git
cd multimodal-rag-gcp

# Install dependencies
pip install -r requirements.txt

# Configure GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Run ingestion
python -m src.ingestion_pipeline --source gs://your-bucket/images/

# Query the RAG system
python -m src.rag_pipeline --query "Find images of damaged car parts"
```

## Key Features

- **Multimodal Search**: Query with text, retrieve relevant images (and vice versa)
- **Gemini Summarization**: Every image is automatically summarized by Gemini Vision
- **BigQuery Vector Search**: Native vector similarity search — no external vector DB needed
- **Incremental Ingestion**: Only processes new/modified images
- **Production Patterns**: Retry logic, logging, batch processing, error handling

## Author

**Chaima Yedes** — Data & AI Architect
- [LinkedIn](https://www.linkedin.com/in/chaima-yedes/)
- yedeschaima5@gmail.com
# Update 2025-06-12
