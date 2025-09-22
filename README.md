# Multimodal RAG Search

A visual search application that lets you query large image databases using natural language or image uploads. Built on GCP with CLIP embeddings, BigQuery vector search, and Gemini for answer generation.

## What it does

- **Text search**: Type "damaged front bumper on silver sedan" and get matching images with an AI-generated summary
- **Image search**: Upload a photo and find visually similar images in the database
- **Tag filtering**: Filter results by category (vehicle, damage type, equipment, etc.)
- **AI answers**: Gemini synthesizes retrieved results into a coherent response

## Architecture

```
                    ┌─────────────────────────────┐
                    │       Streamlit UI           │
                    │  Text query / Image upload   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │       CLIP Encoder           │
                    │  Text or image → 512-dim     │
                    │  embedding vector            │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   BigQuery VECTOR_SEARCH     │
                    │  Cosine similarity on        │
                    │  millions of embeddings      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │       Gemini 1.5 Pro         │
                    │  RAG: context from top-K     │
                    │  results → final answer      │
                    └─────────────────────────────┘
```

### Ingestion pipeline

```
GCS bucket (images)
    │
    ├── CLIP encode → 512-dim embedding
    ├── Gemini Vision → text summary, detected objects, scene
    │
    └── Store in BigQuery:
        embedding | summary | objects | tags | source_path | timestamp
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Embeddings | CLIP ViT-B/32 (512-dim, L2-normalized) |
| Summarization | Google Gemini 1.5 Pro (Vision) |
| Vector store | BigQuery `VECTOR_SEARCH` |
| Object storage | Google Cloud Storage |
| Infrastructure | Terraform, Docker |
| Language | Python 3.10+ |

## Project structure

```
├── app.py                          # Streamlit UI
├── Dockerfile                      # Container build
├── requirements.txt
├── config/
│   └── config.yml                  # GCP project, dataset, model config
├── src/
│   ├── clip_encoder.py             # CLIP embedding (image + text)
│   ├── gemini_summarizer.py        # Gemini Vision image summarization
│   ├── bigquery_vector_store.py    # BigQuery storage + vector search
│   ├── rag_pipeline.py             # End-to-end RAG retrieval + generation
│   └── ingestion_pipeline.py       # Batch ingestion from GCS
├── tests/
│   └── test_pipeline.py
└── docs/
    └── design_decisions.md         # Architecture Decision Records
```

## Getting started

```bash
# Clone
git clone https://github.com/chaimaYS/multimodal-rag-gcp.git
cd multimodal-rag-gcp

# Install
pip install -r requirements.txt

# Configure
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
# Edit config/config.yml with your GCP project and dataset

# Ingest images
python -m src.ingestion_pipeline --source gs://your-bucket/images/

# Launch the UI
streamlit run app.py
```

### Docker

```bash
docker build -t multimodal-rag .
docker run -p 8501:8501 -v ~/.config/gcloud:/root/.config/gcloud multimodal-rag
```

Open http://localhost:8501

## Design decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | CLIP ViT-B/32 | Shared text-image vector space, no fine-tuning needed |
| Vector store | BigQuery | Native `VECTOR_SEARCH`, no external DB, scales to billions |
| Summarization | Gemini 1.5 Pro | GCP-native, strong vision, structured output |
| Similarity metric | Cosine (via dot product on L2-normalized vectors) | Standard for CLIP embeddings |
| UI | Streamlit | Fast to build, supports image display and file upload |

## Author

**Chaima Yedes** — Data & AI Architect
- [LinkedIn](https://www.linkedin.com/in/chaima-yedes/)
