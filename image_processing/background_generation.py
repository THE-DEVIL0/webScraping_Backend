#!/usr/bin/env python3
"""
Background Generation Module
A module for generating backgrounds for product images using OpenAI's DALL·E API.

Author: AI Assistant
Version: 1.0.0
"""

import logging
from typing import List, Optional, Dict, Any
from io import BytesIO
import openai
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackgroundGenerator:
    """A pipeline for generating background images using OpenAI's DALL·E API."""

    def __init__(self, api_key: str, model_name: str = "dall-e-3"):
        self.model_name = model_name
        self._client = None
        self._is_loaded = False
        self._api_key = api_key

    def load_model(self) -> bool:
        """Initialize the OpenAI client.

        Returns:
            True if client is initialized; False otherwise
        """
        try:
            self._client = openai.OpenAI(api_key=self._api_key)
            self._is_loaded = True
            logger.info("OpenAI client initialized for model: %s", self.model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[BytesIO]:
        """Generate background images from a text prompt using OpenAI's DALL·E API.

        Args:
            prompt: Positive text prompt for the background
            num_images: Number of images to generate (DALL·E 3 supports 1 image per request)
            negative_prompt: Not supported by DALL·E 3, included for interface compatibility
            seed: Not supported by DALL·E 3, included for interface compatibility
            kwargs: Additional options (e.g., size, quality)

        Returns:
            List of BytesIO objects containing generated images
        """
        if not self._is_loaded:
            if not self.load_model():
                return []

        logger.info(
            "Generating backgrounds | prompt='%s' | n=%s | seed=%s",
            prompt,
            num_images,
            seed,
        )

        # DALL·E 3 supports only 1 image per request, so we'll loop for num_images
        results: List[BytesIO] = []
        for i in range(max(1, min(num_images, 10))):  # Cap at 10 to avoid API abuse
            try:
                # Configure generation parameters
                generation_params = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "n": 1,  # DALL·E 3 supports only 1 image per request
                    "size": kwargs.get("size", "1024x1024"),
                    "quality": kwargs.get("quality", "standard"),
                    "response_format": "url"
                }

                # Generate image
                response = self._client.images.generate(**generation_params)
                image_url = response.data[0].url

                # Fetch image into BytesIO
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_bytes = BytesIO(image_response.content)
                    results.append(image_bytes)
                    logger.info(f"Generated image {i+1} fetched to BytesIO")
                else:
                    logger.warning(f"Failed to fetch image from {image_url}")

            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                continue

        return results

def main():
    """Simple CLI for the background generator."""
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Background Generation with OpenAI DALL·E")
    parser.add_argument("prompt", help="Text prompt for background generation")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--negative", dest="negative", default=None, help="Negative prompt (not supported)")
    parser.add_argument("-o", "--output", default="images/generated_backgrounds", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (not supported)")
    parser.add_argument("--model", default="dall-e-3", help="Model name (default: dall-e-3)")
    parser.add_argument("--size", default="1024x1024", help="Image size (e.g., 1024x1024)")
    parser.add_argument("--quality", default="standard", choices=["standard", "hd"], help="Image quality")

    args = parser.parse_args()

    # Initialize with the provided API key
    gen = BackgroundGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=args.model
    )
    print(f"Using generator: {gen}")
    image_bytes_list = gen.generate(
        prompt=args.prompt,
        num_images=args.num_images,
        negative_prompt=args.negative,
        seed=args.seed,
        size=args.size,
        quality=args.quality
    )

    # Save images for CLI usage
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, image_bytes in enumerate(image_bytes_list, 1):
        file_path = output_dir / f"bg_{i:03d}.png"
        with open(file_path, "wb") as f:
            image_bytes.seek(0)
            f.write(image_bytes.getvalue())
        saved_paths.append(file_path)

    if saved_paths:
        print("Generated:")
        for p in saved_paths:
            print(p)
    else:
        print("No images generated")

if __name__ == "__main__":
    main()