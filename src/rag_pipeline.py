"""
RAG Pipeline — Multimodal Retrieval-Augmented Generation.

End-to-end pipeline:
1. Encode user query with CLIP
2. Search BigQuery for similar images
3. Build context from retrieved summaries
4. Generate final answer with Gemini
"""

import google.generativeai as genai
from src.clip_encoder import CLIPEncoder
from src.bigquery_vector_store import BigQueryVectorStore
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Container for RAG query results."""
    query: str
    answer: str
    retrieved_images: List[dict]
    context_used: str
    num_results: int


class MultimodalRAGPipeline:
    """
    Multimodal RAG system that combines:
    - CLIP for cross-modal retrieval (text → image)
    - BigQuery for vector storage and search
    - Gemini for answer generation with retrieved context
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        gemini_api_key: str,
        table_id: str = "multimodal_embeddings",
        gemini_model: str = "gemini-1.5-pro",
    ):
        self.encoder = CLIPEncoder()
        self.vector_store = BigQueryVectorStore(project_id, dataset_id, table_id)

        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel(gemini_model)

        logger.info("Multimodal RAG Pipeline initialized")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None,
        include_images: bool = True,
    ) -> RAGResult:
        """
        Execute a RAG query:
        1. Encode query text with CLIP
        2. Search for similar images in BigQuery
        3. Build context from retrieved summaries
        4. Generate answer with Gemini
        
        Args:
            query_text: Natural language query
            top_k: Number of images to retrieve
            filter_tags: Optional tag filter
            include_images: Whether to include image paths in response
            
        Returns:
            RAGResult with answer and supporting images
        """
        logger.info(f"RAG query: '{query_text}' (top_k={top_k})")

        # Step 1: Encode query
        query_embedding = self.encoder.encode_text(query_text)
        logger.info("Query encoded with CLIP")

        # Step 2: Retrieve similar images from BigQuery
        retrieved = self.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_tags=filter_tags,
        )
        logger.info(f"Retrieved {len(retrieved)} results from BigQuery")

        if not retrieved:
            return RAGResult(
                query=query_text,
                answer="No relevant images found for your query.",
                retrieved_images=[],
                context_used="",
                num_results=0,
            )

        # Step 3: Build context from retrieved summaries
        context = self._build_context(retrieved)

        # Step 4: Generate answer with Gemini
        answer = self._generate_answer(query_text, context)

        return RAGResult(
            query=query_text,
            answer=answer,
            retrieved_images=retrieved if include_images else [],
            context_used=context,
            num_results=len(retrieved),
        )

    def _build_context(self, retrieved: List[dict]) -> str:
        """
        Build a structured context string from retrieved results.
        This context is injected into the Gemini prompt.
        """
        context_parts = []

        for i, result in enumerate(retrieved, 1):
            part = f"""
Image {i} (similarity: {result['similarity_score']:.3f}):
- Source: {result['source_path']}
- Summary: {result['summary']}
- Objects: {', '.join(result['objects']) if result['objects'] else 'N/A'}
- Scene: {result['scene'] or 'N/A'}
- Tags: {', '.join(result['tags']) if result['tags'] else 'N/A'}
- Text in image: {result['text_in_image'] or 'None'}
"""
            context_parts.append(part.strip())

        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate a comprehensive answer using Gemini with retrieved context."""
        prompt = f"""You are a multimodal AI assistant. A user has asked a question and 
we have retrieved the most relevant images from our database.

USER QUERY: {query}

RETRIEVED IMAGE CONTEXT:
{context}

Based on the retrieved images and their descriptions, provide a helpful and accurate 
answer to the user's query. Reference specific images when relevant. If the retrieved 
images don't fully answer the question, acknowledge the limitations.

ANSWER:"""

        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            # Fallback: return summaries directly
            summaries = [r["summary"] for r in self.vector_store.vector_search(
                self.encoder.encode_text(query), top_k=3
            ) if r.get("summary")]
            return f"Based on retrieved images: {' '.join(summaries)}" if summaries else "Unable to generate answer."


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Query the Multimodal RAG system")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--config", type=str, default="config/config.yml", help="Config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = MultimodalRAGPipeline(
        project_id=config["gcp"]["project_id"],
        dataset_id=config["gcp"]["dataset_id"],
        gemini_api_key=config["gemini"]["api_key"],
    )

    result = pipeline.query(args.query, top_k=args.top_k)

    print(f"\nQuery: {result.query}")
    print(f"Results: {result.num_results}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nRetrieved images:")
    for img in result.retrieved_images:
        print(f"  - {img['source_path']} (score: {img['similarity_score']:.3f})")
