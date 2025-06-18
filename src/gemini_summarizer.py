"""
Gemini Vision Summarizer — Generates natural language descriptions of images.

Uses Google Gemini 1.5 Pro (Vision) to analyze images and produce
structured summaries for the RAG knowledge base.
"""

import google.generativeai as genai
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional
import time
import logging
import json

logger = logging.getLogger(__name__)


class GeminiSummarizer:
    """
    Image summarization using Google Gemini Vision API.
    
    Generates structured descriptions that are stored alongside
    CLIP embeddings in BigQuery for enriched RAG retrieval.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        logger.info(f"Gemini model initialized: {model_name}")

    def summarize_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate a structured summary of an image using Gemini Vision.
        
        Args:
            image: File path or PIL Image
            prompt: Custom prompt (uses default if None)
            max_retries: Number of retry attempts on failure
            
        Returns:
            dict with keys: summary, objects, scene, tags, raw_response
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        if prompt is None:
            prompt = self._default_prompt()

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    ),
                )

                parsed = self._parse_response(response.text)
                logger.info(f"Image summarized successfully")
                return parsed

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for image summarization")
                    return self._empty_result(str(e))

    def summarize_images_batch(
        self,
        images: List[Union[str, Path]],
        rate_limit_delay: float = 1.0,
    ) -> List[dict]:
        """
        Summarize multiple images with rate limiting.
        
        Args:
            images: List of image file paths
            rate_limit_delay: Seconds to wait between API calls
            
        Returns:
            List of summary dicts
        """
        results = []

        for i, image_path in enumerate(images):
            logger.info(f"Summarizing image {i + 1}/{len(images)}: {image_path}")
            result = self.summarize_image(image_path)
            result["source_path"] = str(image_path)
            results.append(result)

            # Rate limiting to avoid API quota issues
            if i < len(images) - 1:
                time.sleep(rate_limit_delay)

        logger.info(f"Batch summarization complete: {len(results)} images processed")
        return results

    def _default_prompt(self) -> str:
        """Default structured prompt for image analysis."""
        return """Analyze this image and provide a structured description in the following JSON format:

{
    "summary": "A 2-3 sentence description of the image content",
    "objects": ["list", "of", "main", "objects", "detected"],
    "scene": "Brief description of the scene or setting",
    "tags": ["relevant", "searchable", "tags"],
    "dominant_colors": ["color1", "color2"],
    "text_in_image": "Any visible text in the image, or null if none"
}

Respond ONLY with the JSON object, no additional text."""

    def _parse_response(self, response_text: str) -> dict:
        """Parse Gemini response into structured dict."""
        try:
            # Clean response — remove markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed = json.loads(cleaned.strip())
            parsed["raw_response"] = response_text
            parsed["parse_status"] = "success"
            return parsed

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from Gemini response, using raw text")
            return {
                "summary": response_text[:500],
                "objects": [],
                "scene": "",
                "tags": [],
                "dominant_colors": [],
                "text_in_image": None,
                "raw_response": response_text,
                "parse_status": "fallback_raw_text",
            }

    def _empty_result(self, error_msg: str) -> dict:
        """Return empty result on complete failure."""
        return {
            "summary": "",
            "objects": [],
            "scene": "",
            "tags": [],
            "dominant_colors": [],
            "text_in_image": None,
            "raw_response": "",
            "parse_status": f"error: {error_msg}",
        }
