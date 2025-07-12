"""
Ingestion Pipeline — Batch processing for image ingestion.

Orchestrates the full flow:
1. List images from GCS
2. Generate CLIP embeddings (batch)
3. Generate Gemini summaries (batch with rate limiting)
4. Store everything in BigQuery
"""

from google.cloud import storage
from src.clip_encoder import CLIPEncoder
from src.gemini_summarizer import GeminiSummarizer
from src.bigquery_vector_store import BigQueryVectorStore
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import hashlib
import tempfile
import logging
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for multimodal data.
    
    Downloads images from GCS → encodes with CLIP → summarizes with Gemini
    → stores embeddings + metadata in BigQuery.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        gemini_api_key: str,
        table_id: str = "multimodal_embeddings",
    ):
        self.encoder = CLIPEncoder()
        self.summarizer = GeminiSummarizer(api_key=gemini_api_key)
        self.vector_store = BigQueryVectorStore(project_id, dataset_id, table_id)
        self.gcs_client = storage.Client(project=project_id)

        # Ensure table exists
        self.vector_store.create_table_if_not_exists(embedding_dim=self.encoder.embedding_dim)

        logger.info("Ingestion pipeline initialized")

    def ingest_from_gcs(
        self,
        bucket_name: str,
        prefix: str = "",
        file_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
        batch_size: int = 32,
        rate_limit_delay: float = 1.0,
    ) -> dict:
        """
        Ingest images from a GCS bucket.
        
        Args:
            bucket_name: GCS bucket name
            prefix: Path prefix filter
            file_extensions: Allowed image extensions
            batch_size: CLIP encoding batch size
            rate_limit_delay: Delay between Gemini API calls
            
        Returns:
            dict with ingestion statistics
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting ingestion from gs://{bucket_name}/{prefix}")

        # Step 1: List images in GCS
        image_blobs = self._list_images(bucket_name, prefix, file_extensions)
        logger.info(f"Found {len(image_blobs)} images to process")

        if not image_blobs:
            return {"status": "no_images", "count": 0}

        # Step 2: Process in batches
        total_processed = 0
        total_errors = 0

        for i in range(0, len(image_blobs), batch_size):
            batch_blobs = image_blobs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(image_blobs) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_blobs)} images)")

            try:
                records = self._process_batch(
                    bucket_name, batch_blobs, rate_limit_delay
                )
                self.vector_store.upsert_embeddings(records)
                total_processed += len(records)
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                total_errors += len(batch_blobs)

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        stats = {
            "status": "complete",
            "total_images": len(image_blobs),
            "processed": total_processed,
            "errors": total_errors,
            "elapsed_seconds": round(elapsed, 2),
            "images_per_second": round(total_processed / elapsed, 2) if elapsed > 0 else 0,
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def _list_images(self, bucket_name: str, prefix: str, extensions: tuple) -> list:
        """List image files in GCS bucket."""
        bucket = self.gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        image_blobs = [
            blob.name for blob in blobs
            if blob.name.lower().endswith(extensions)
        ]
        return image_blobs

    def _process_batch(
        self,
        bucket_name: str,
        blob_names: List[str],
        rate_limit_delay: float,
    ) -> List[dict]:
        """
        Process a batch of images:
        1. Download from GCS to temp dir
        2. Encode with CLIP
        3. Summarize with Gemini
        4. Build records for BigQuery
        """
        records = []
        bucket = self.gcs_client.bucket(bucket_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download images
            local_paths = []
            for blob_name in blob_names:
                local_path = Path(tmpdir) / Path(blob_name).name
                blob = bucket.blob(blob_name)
                blob.download_to_filename(str(local_path))
                local_paths.append((blob_name, local_path))

            # Batch encode with CLIP
            images_for_clip = [str(lp) for _, lp in local_paths]
            embeddings = self.encoder.encode_images_batch(images_for_clip)

            # Summarize each image with Gemini (with rate limiting)
            for idx, (blob_name, local_path) in enumerate(local_paths):
                # Generate summary
                summary_result = self.summarizer.summarize_image(local_path)

                # Build record
                image_id = self._generate_image_id(bucket_name, blob_name)
                record = {
                    "image_id": image_id,
                    "source_path": f"gs://{bucket_name}/{blob_name}",
                    "embedding": embeddings[idx],
                    "summary": summary_result.get("summary", ""),
                    "objects": summary_result.get("objects", []),
                    "scene": summary_result.get("scene", ""),
                    "tags": summary_result.get("tags", []),
                    "text_in_image": summary_result.get("text_in_image"),
                }
                records.append(record)

        return records

    @staticmethod
    def _generate_image_id(bucket_name: str, blob_name: str) -> str:
        """Generate a deterministic unique ID for an image."""
        raw = f"{bucket_name}/{blob_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest images into the multimodal RAG system")
    parser.add_argument("--source", type=str, required=True, help="GCS path (gs://bucket/prefix)")
    parser.add_argument("--config", type=str, default="config/config.yml", help="Config file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP encoding")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Parse GCS path
    gcs_parts = args.source.replace("gs://", "").split("/", 1)
    bucket = gcs_parts[0]
    prefix = gcs_parts[1] if len(gcs_parts) > 1 else ""

    pipeline = IngestionPipeline(
        project_id=config["gcp"]["project_id"],
        dataset_id=config["gcp"]["dataset_id"],
        gemini_api_key=config["gemini"]["api_key"],
    )

    stats = pipeline.ingest_from_gcs(
        bucket_name=bucket,
        prefix=prefix,
        batch_size=args.batch_size,
    )

    print(f"\nIngestion Stats: {stats}")
# Upload handler
