# Design Decisions — Multimodal RAG System

## DD-001: BigQuery as Vector Store (vs. Pinecone/Qdrant/FAISS)

**Decision:** Use BigQuery's native `VECTOR_SEARCH` instead of a dedicated vector database.

| Option | Pros | Cons |
|--------|------|------|
| **BigQuery (chosen)** | No extra infra, SQL-native, scales to billions, joins with other data | Newer feature, latency higher than dedicated DBs |
| Pinecone | Purpose-built, low latency | Extra cost, separate system, vendor lock-in |
| Qdrant | Open source, fast | Requires hosting and management |
| FAISS | Fast in-memory search | No persistence, no SQL, single-machine limit |

**Rationale:** For an enterprise system already on GCP, keeping embeddings in BigQuery reduces infrastructure complexity. The metadata (summaries, tags, objects) lives alongside the vectors, enabling rich filtered searches in a single query. BigQuery vector index (IVF) provides acceptable latency for batch/near-real-time use cases.

---

## DD-002: CLIP for Embeddings (vs. other models)

**Decision:** Use OpenAI CLIP (ViT-B/32) for both image and text embeddings.

**Rationale:**
- CLIP maps images and text into the **same vector space**, enabling cross-modal search (query with text, retrieve images)
- Pre-trained on 400M image-text pairs — strong zero-shot performance
- 512-dimensional embeddings — compact and efficient for storage
- ViT-B/32 is a good balance between quality and speed

**Alternative considered:** Google's multimodal embeddings via Vertex AI. Rejected because CLIP is open-source, more portable, and easier to reproduce.

---

## DD-003: Gemini for Summarization (vs. GPT-4V / Claude)

**Decision:** Use Google Gemini 1.5 Pro for image summarization.

**Rationale:**
- Native GCP integration — same project, same billing, same IAM
- Strong vision capabilities with structured output support
- Lower latency when running in same GCP region
- Cost-effective for batch processing with rate limiting

**Structured output strategy:** Prompt Gemini to return JSON with predefined schema. Parse with fallback to raw text if JSON parsing fails. This ensures consistent metadata structure in BigQuery.

---

## DD-004: Separation of Embedding and Summarization

**Decision:** Generate CLIP embeddings and Gemini summaries as **separate, independent steps**.

**Rationale:**
- CLIP embeddings are deterministic and fast (batch GPU processing)
- Gemini summaries are non-deterministic and rate-limited (sequential API calls)
- If Gemini fails, the embedding is still stored — the record is still searchable
- Summaries can be regenerated independently without re-encoding

---

## DD-005: L2 Normalization of Embeddings

**Decision:** All embeddings are L2-normalized before storage.

**Rationale:**
- Cosine similarity = dot product when vectors are unit-length
- Dot product is faster to compute than full cosine similarity
- BigQuery `VECTOR_SEARCH` with `COSINE` distance works optimally with normalized vectors
- Consistent similarity scores (always between -1 and 1)
