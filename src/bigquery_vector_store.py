"""
BigQuery Vector Store — Storage and retrieval layer for multimodal embeddings.

Uses BigQuery's native VECTOR_SEARCH function for scalable similarity search
without needing an external vector database.
"""

from google.cloud import bigquery
import numpy as np
import json
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BigQueryVectorStore:
    """
    Vector storage and retrieval using BigQuery.
    
    Stores CLIP embeddings alongside Gemini summaries and metadata.
    Uses BigQuery VECTOR_SEARCH for efficient similarity queries.
    """

    def __init__(self, project_id: str, dataset_id: str, table_id: str = "multimodal_embeddings"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        logger.info(f"BigQuery Vector Store initialized: {self.full_table_id}")

    def create_table_if_not_exists(self, embedding_dim: int = 512) -> None:
        """
        Create the embeddings table with vector column.
        Uses BigQuery FLOAT64 ARRAY for embedding storage.
        """
        schema = [
            bigquery.SchemaField("image_id", "STRING", mode="REQUIRED",
                                 description="Unique identifier for the image"),
            bigquery.SchemaField("source_path", "STRING", mode="REQUIRED",
                                 description="GCS path to the original image"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED",
                                 description=f"CLIP embedding vector ({embedding_dim}D)"),
            bigquery.SchemaField("summary", "STRING", mode="NULLABLE",
                                 description="Gemini-generated image summary"),
            bigquery.SchemaField("objects", "STRING", mode="REPEATED",
                                 description="Detected objects in the image"),
            bigquery.SchemaField("scene", "STRING", mode="NULLABLE",
                                 description="Scene description"),
            bigquery.SchemaField("tags", "STRING", mode="REPEATED",
                                 description="Searchable tags"),
            bigquery.SchemaField("text_in_image", "STRING", mode="NULLABLE",
                                 description="OCR text detected in image"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                                 description="Ingestion timestamp"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED",
                                 description="Last update timestamp"),
        ]

        table_ref = bigquery.Table(self.full_table_id, schema=schema)

        try:
            self.client.get_table(self.full_table_id)
            logger.info(f"Table {self.full_table_id} already exists")
        except Exception:
            table = self.client.create_table(table_ref)
            logger.info(f"Created table {table.full_table_id}")

            # Create vector index for fast similarity search
            self._create_vector_index(embedding_dim)

    def _create_vector_index(self, embedding_dim: int) -> None:
        """Create a vector index on the embedding column for fast search."""
        index_query = f"""
        CREATE VECTOR INDEX IF NOT EXISTS idx_embedding
        ON `{self.full_table_id}`(embedding)
        OPTIONS (
            index_type = 'IVF',
            distance_type = 'COSINE',
            ivf_options = '{{"num_lists": 100}}'
        )
        """
        try:
            self.client.query(index_query).result()
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Vector index creation note: {e}")

    def upsert_embeddings(self, records: List[dict]) -> int:
        """
        Insert or update embedding records.
        
        Args:
            records: List of dicts with keys:
                - image_id, source_path, embedding, summary,
                  objects, scene, tags, text_in_image
                  
        Returns:
            Number of rows inserted
        """
        now = datetime.utcnow().isoformat()

        rows_to_insert = []
        for record in records:
            row = {
                "image_id": record["image_id"],
                "source_path": record["source_path"],
                "embedding": record["embedding"].tolist() if isinstance(record["embedding"], np.ndarray) else record["embedding"],
                "summary": record.get("summary", ""),
                "objects": record.get("objects", []),
                "scene": record.get("scene", ""),
                "tags": record.get("tags", []),
                "text_in_image": record.get("text_in_image"),
                "created_at": now,
                "updated_at": now,
            }
            rows_to_insert.append(row)

        errors = self.client.insert_rows_json(self.full_table_id, rows_to_insert)

        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise RuntimeError(f"Failed to insert rows: {errors}")

        logger.info(f"Inserted {len(rows_to_insert)} rows into {self.full_table_id}")
        return len(rows_to_insert)

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Perform vector similarity search using BigQuery VECTOR_SEARCH.
        
        Args:
            query_embedding: Query vector (512D), L2-normalized
            top_k: Number of results to return
            filter_tags: Optional tag filter
            
        Returns:
            List of dicts with image_id, source_path, summary, score
        """
        embedding_str = json.dumps(query_embedding.tolist())

        # Build optional WHERE clause for tag filtering
        tag_filter = ""
        if filter_tags:
            tag_conditions = " OR ".join([f"'{tag}' IN UNNEST(base.tags)" for tag in filter_tags])
            tag_filter = f"WHERE ({tag_conditions})"

        query = f"""
        SELECT
            base.image_id,
            base.source_path,
            base.summary,
            base.objects,
            base.scene,
            base.tags,
            base.text_in_image,
            distance
        FROM
            VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                'embedding',
                (SELECT {embedding_str} AS embedding),
                top_k => {top_k},
                distance_type => 'COSINE'
            )
        {tag_filter}
        ORDER BY distance ASC
        LIMIT {top_k}
        """

        results = self.client.query(query).result()

        output = []
        for row in results:
            output.append({
                "image_id": row.image_id,
                "source_path": row.source_path,
                "summary": row.summary,
                "objects": list(row.objects) if row.objects else [],
                "scene": row.scene,
                "tags": list(row.tags) if row.tags else [],
                "text_in_image": row.text_in_image,
                "similarity_score": 1 - row.distance,  # Convert distance to similarity
            })

        logger.info(f"Vector search returned {len(output)} results")
        return output

    def get_stats(self) -> dict:
        """Get table statistics."""
        query = f"""
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT image_id) AS unique_images,
            MIN(created_at) AS earliest_record,
            MAX(updated_at) AS latest_record,
            COUNTIF(summary IS NOT NULL AND summary != '') AS summarized_count
        FROM `{self.full_table_id}`
        """
        result = list(self.client.query(query).result())[0]
        return {
            "total_records": result.total_records,
            "unique_images": result.unique_images,
            "earliest_record": str(result.earliest_record),
            "latest_record": str(result.latest_record),
            "summarized_count": result.summarized_count,
        }

    def delete_by_image_id(self, image_id: str) -> None:
        """Delete all records for a given image_id."""
        query = f"DELETE FROM `{self.full_table_id}` WHERE image_id = '{image_id}'"
        self.client.query(query).result()
        logger.info(f"Deleted records for image_id: {image_id}")
