"""
Unit tests for the Multimodal RAG pipeline components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image


class TestCLIPEncoder:
    """Tests for CLIP embedding generation."""

    @pytest.fixture
    def encoder(self):
        from src.clip_encoder import CLIPEncoder
        return CLIPEncoder(device="cpu")

    def test_embedding_dimension(self, encoder):
        """Embeddings should be 512-dimensional."""
        assert encoder.embedding_dim == 512

    def test_image_embedding_shape(self, encoder):
        """Image embedding should have correct shape."""
        test_image = Image.new("RGB", (224, 224), color="red")
        embedding = encoder.encode_image(test_image)
        assert embedding.shape == (512,)

    def test_text_embedding_shape(self, encoder):
        """Text embedding should have correct shape."""
        embedding = encoder.encode_text("a photo of a car")
        assert embedding.shape == (512,)

    def test_embedding_normalized(self, encoder):
        """Embeddings should be L2-normalized (unit vectors)."""
        embedding = encoder.encode_text("test query")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Embedding norm should be ~1.0, got {norm}"

    def test_similar_texts_closer(self, encoder):
        """Semantically similar texts should have higher similarity."""
        emb_car = encoder.encode_text("a red sports car")
        emb_vehicle = encoder.encode_text("a fast automobile")
        emb_cat = encoder.encode_text("a fluffy cat sleeping")

        sim_car_vehicle = np.dot(emb_car, emb_vehicle)
        sim_car_cat = np.dot(emb_car, emb_cat)

        assert sim_car_vehicle > sim_car_cat, "Car and vehicle should be more similar than car and cat"

    def test_batch_encoding(self, encoder):
        """Batch encoding should return correct number of embeddings."""
        images = [Image.new("RGB", (224, 224), color=c) for c in ["red", "blue", "green"]]
        embeddings = encoder.encode_images_batch(images, batch_size=2)
        assert embeddings.shape == (3, 512)

    def test_cross_modal_similarity(self, encoder):
        """Text query about red should be more similar to red image than blue."""
        red_image = Image.new("RGB", (224, 224), color="red")
        blue_image = Image.new("RGB", (224, 224), color="blue")

        text_emb = encoder.encode_text("a solid red image")
        red_emb = encoder.encode_image(red_image)
        blue_emb = encoder.encode_image(blue_image)

        sim_red = np.dot(text_emb, red_emb)
        sim_blue = np.dot(text_emb, blue_emb)

        # Red text should be closer to red image
        assert sim_red > sim_blue


class TestGeminiSummarizer:
    """Tests for Gemini summarization (mocked API)."""

    def test_parse_valid_json(self):
        from src.gemini_summarizer import GeminiSummarizer

        summarizer = GeminiSummarizer.__new__(GeminiSummarizer)
        response = '{"summary": "A car on a road", "objects": ["car", "road"], "scene": "highway", "tags": ["vehicle"], "dominant_colors": ["gray"], "text_in_image": null}'
        result = summarizer._parse_response(response)

        assert result["summary"] == "A car on a road"
        assert "car" in result["objects"]
        assert result["parse_status"] == "success"

    def test_parse_json_with_markdown(self):
        from src.gemini_summarizer import GeminiSummarizer

        summarizer = GeminiSummarizer.__new__(GeminiSummarizer)
        response = '```json\n{"summary": "A building", "objects": ["building"], "scene": "urban", "tags": ["architecture"], "dominant_colors": ["white"], "text_in_image": null}\n```'
        result = summarizer._parse_response(response)

        assert result["summary"] == "A building"
        assert result["parse_status"] == "success"

    def test_parse_invalid_json_fallback(self):
        from src.gemini_summarizer import GeminiSummarizer

        summarizer = GeminiSummarizer.__new__(GeminiSummarizer)
        response = "This is just plain text, not JSON"
        result = summarizer._parse_response(response)

        assert result["parse_status"] == "fallback_raw_text"
        assert result["summary"] == response

    def test_empty_result_on_error(self):
        from src.gemini_summarizer import GeminiSummarizer

        summarizer = GeminiSummarizer.__new__(GeminiSummarizer)
        result = summarizer._empty_result("API timeout")

        assert result["summary"] == ""
        assert "error" in result["parse_status"]


class TestBigQueryVectorStore:
    """Tests for BigQuery vector store (mocked client)."""

    def test_generate_image_id(self):
        from src.ingestion_pipeline import IngestionPipeline

        id1 = IngestionPipeline._generate_image_id("bucket", "path/image.jpg")
        id2 = IngestionPipeline._generate_image_id("bucket", "path/image.jpg")
        id3 = IngestionPipeline._generate_image_id("bucket", "path/other.jpg")

        assert id1 == id2, "Same input should produce same ID"
        assert id1 != id3, "Different input should produce different ID"
        assert len(id1) == 16, "ID should be 16 chars"
# Eval metrics
