"""
CLIP Encoder — Generates embeddings for images and text.

Uses OpenAI CLIP (ViT-B/32) to produce 512-dimensional embeddings
that share the same vector space, enabling cross-modal similarity search.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    Multimodal encoder using CLIP for image and text embeddings.
    
    Embeddings are L2-normalized so cosine similarity = dot product,
    which is efficient for BigQuery VECTOR_SEARCH.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim  # 512
        logger.info(f"CLIP loaded. Embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode a single image into a CLIP embedding vector.
        
        Args:
            image: File path, Path object, or PIL Image
            
        Returns:
            np.ndarray of shape (512,), L2-normalized
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)

        # L2 normalize for cosine similarity
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def encode_images_batch(self, images: List[Union[str, Path, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple images in batches for efficiency.
        
        Args:
            images: List of file paths or PIL Images
            batch_size: Number of images per batch
            
        Returns:
            np.ndarray of shape (N, 512)
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load images
            pil_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    try:
                        pil_images.append(Image.open(img).convert("RGB"))
                    except Exception as e:
                        logger.warning(f"Failed to load image {img}: {e}")
                        pil_images.append(Image.new("RGB", (224, 224)))  # placeholder
                else:
                    pil_images.append(img)

            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())
            logger.info(f"Encoded batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a CLIP embedding vector.
        
        Args:
            text: Search query string
            
        Returns:
            np.ndarray of shape (512,), L2-normalized
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        embedding = self.model.get_text_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode multiple text queries in batches.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            
        Returns:
            np.ndarray of shape (N, 512)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Encode an image from raw bytes (e.g., uploaded file)."""
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.encode_image(image)

    def compute_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and candidate embeddings.
        Since embeddings are L2-normalized, cosine similarity = dot product.
        
        Args:
            query_embedding: shape (512,)
            candidate_embeddings: shape (N, 512)
            
        Returns:
            np.ndarray of similarity scores, shape (N,)
        """
        return np.dot(candidate_embeddings, query_embedding)
