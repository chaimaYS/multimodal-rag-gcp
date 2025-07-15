# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Multimodal RAG — Interactive Demo
#
# This notebook demonstrates the end-to-end multimodal RAG pipeline:
# 1. Encode images with CLIP
# 2. Summarize images with Gemini
# 3. Store in BigQuery
# 4. Query with natural language

# %%
import sys
sys.path.append("..")

from src.clip_encoder import CLIPEncoder
from src.gemini_summarizer import GeminiSummarizer
from src.bigquery_vector_store import BigQueryVectorStore
from src.rag_pipeline import MultimodalRAGPipeline
from PIL import Image
import numpy as np

# %% [markdown]
# ## 1. Initialize Components

# %%
# CLIP Encoder
encoder = CLIPEncoder(device="cpu")
print(f"CLIP loaded. Embedding dim: {encoder.embedding_dim}")

# %% [markdown]
# ## 2. Test CLIP Embeddings

# %%
# Encode a text query
query = "a damaged car bumper"
text_embedding = encoder.encode_text(query)
print(f"Text embedding shape: {text_embedding.shape}")
print(f"Text embedding norm: {np.linalg.norm(text_embedding):.4f}")  # Should be ~1.0

# %%
# Encode a test image
test_image = Image.new("RGB", (224, 224), color="blue")
image_embedding = encoder.encode_image(test_image)
print(f"Image embedding shape: {image_embedding.shape}")

# %%
# Cross-modal similarity
similarity = np.dot(text_embedding, image_embedding)
print(f"Text-Image similarity: {similarity:.4f}")

# %% [markdown]
# ## 3. Test Gemini Summarization

# %%
# Note: Requires GEMINI_API_KEY environment variable
# summarizer = GeminiSummarizer(api_key="YOUR_API_KEY")
# result = summarizer.summarize_image(test_image)
# print(result)

# %% [markdown]
# ## 4. Query the RAG System

# %%
# Note: Requires GCP project with BigQuery dataset
# pipeline = MultimodalRAGPipeline(
#     project_id="your-project",
#     dataset_id="multimodal_rag",
#     gemini_api_key="YOUR_API_KEY",
# )
#
# result = pipeline.query("Find images of damaged car parts", top_k=3)
# print(f"Answer: {result.answer}")
# print(f"Retrieved {result.num_results} images")

# %% [markdown]
# ## 5. Similarity Search Examples

# %%
# Example: Compare multiple queries against an image
queries = [
    "a red sports car",
    "a broken windshield",
    "a sunny beach",
    "a factory assembly line",
]

query_embeddings = encoder.encode_texts_batch(queries)

# Compute similarities against a test image
for i, query in enumerate(queries):
    sim = np.dot(query_embeddings[i], image_embedding)
    print(f"  '{query}' → similarity: {sim:.4f}")
