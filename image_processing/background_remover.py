#!/usr/bin/env python3
"""
Background Removal Module
A production-ready background removal tool using Hugging Face RMBG model.
Integrated with Amazon Image Scraper for seamless workflow.

Version: 1.0.3
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
from PIL import Image
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logger = logging.getLogger(__name__)

class BackgroundRemover:
    """
    Professional background removal tool using AI models.
    """
    
    def __init__(self, model_name: str = "briaai/RMBG-1.4", max_workers: int = 2):
        """
        Initialize the Background Remover.
        
        Args:
            model_name: Hugging Face model to use for background removal
            max_workers: Maximum concurrent processing threads
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.remover = None
        self._model_loaded = False
        
        # Performance tracking
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        print(f"BackgroundRemover: Initialized with model {model_name}, max_workers={max_workers}")
        
    def load_model(self) -> bool:
        """
        Load the background removal model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading background removal model: {self.model_name}")
            print(f"BackgroundRemover.load_model: Loading {self.model_name}")
            
            from transformers import pipeline
            
            self.remover = pipeline(
                "image-segmentation",
                model=self.model_name,
                trust_remote_code=True
            )
            
            self._model_loaded = True
            logger.info("Background removal model loaded successfully")
            print("BackgroundRemover.load_model: Model loaded successfully")
            return True
            
        except ImportError as e:
            missing_deps = []
            error_msg = str(e).lower()
            
            if 'transformers' in error_msg:
                missing_deps.append('transformers')
            if 'torch' in error_msg:
                missing_deps.append('torch')
            if 'skimage' in error_msg or 'scikit-image' in error_msg:
                missing_deps.append('scikit-image')
            
            if missing_deps:
                logger.error(f"Missing dependencies: {', '.join(missing_deps)}. Run: pip install {' '.join(missing_deps)}")
                print(f"BackgroundRemover.load_model: Missing dependencies: {', '.join(missing_deps)}")
            else:
                logger.error(f"Import error: {e}. Run: pip install transformers torch scikit-image")
                print(f"BackgroundRemover.load_model: Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load background removal model: {e}")
            print(f"BackgroundRemover.load_model: Failed to load model: {e}")
            return False
    
    def remove_background(self, image: Image.Image, add_white_background: bool = False) -> Optional[BytesIO]:
        """
        Remove background from a single PIL Image.
        
        Args:
            image: PIL Image object to process
            add_white_background: If True, composite onto white background
            
        Returns:
            BytesIO containing the processed image if successful, None otherwise
        """
        if not self._model_loaded:
            if not self.load_model():
                print(f"BackgroundRemover.remove_background: Failed to load model")
                return None
        
        try:
            print(f"BackgroundRemover.remove_background: Processing image")
            
            # Process image
            logger.debug("Processing image")
            result_image = self.remover(image)
            print(f"BackgroundRemover.remove_background: Background removed")
            
            # Add white background if requested
            if add_white_background and result_image.mode == 'RGBA':
                white_bg = Image.new("RGB", result_image.size, (255, 255, 255))
                mask = result_image.split()[-1]
                white_bg.paste(result_image, (0, 0), mask=mask)
                result_image = white_bg
                print(f"BackgroundRemover.remove_background: Added white background")
            
            # Save to BytesIO
            output = BytesIO()
            if add_white_background:
                result_image.save(output, format="JPEG", quality=95)
            else:
                result_image.save(output, format="PNG")
            output.seek(0)
            
            logger.debug("Background removed and saved to BytesIO")
            print(f"BackgroundRemover.remove_background: Saved to BytesIO")
            return output
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to remove background: {e}")
            print(f"BackgroundRemover.remove_background: Failed: {e}")
            return None
        finally:
            self.processed_count += 1
    
    def process_selected_images(self, image_objects: Optional[List[Tuple[str, Image.Image]]] = None,
                               image_paths: Optional[List[Union[str, Path]]] = None,
                               add_white_background: bool = False,
                               progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, any]:
        """
        Process a specific list of image objects or paths.
        
        Args:
            image_objects: List of (url, PIL.Image) tuples to process
            image_paths: List of image file paths to process (for backward compatibility)
            add_white_background: If True, composite onto white background
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing statistics and results (url, BytesIO) or paths
        """
        if image_objects and image_paths:
            raise ValueError("Cannot provide both image_objects and image_paths")
        if not image_objects and not image_paths:
            raise ValueError("Must provide either image_objects or image_paths")
        
        if image_objects:
            items = [(url, img) for url, img in image_objects]
            item_type = "image_objects"
            total_items = len(image_objects)
            print(f"BackgroundRemover.process_selected_images: Starting with {total_items} {item_type}: {[url for url, _ in items]}")
        else:
            items = [(str(path), Image.open(path).convert("RGB")) for path in map(Path, image_paths)]
            item_type = "image_paths"
            total_items = len(image_paths)
            print(f"BackgroundRemover.process_selected_images: Starting with {total_items} {item_type}: {image_paths}")
        
        logger.info(f"Processing {total_items} {item_type}")
        
        # Reset counters
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {}
            for identifier, image in items:
                future = executor.submit(
                    self.remove_background,
                    image,
                    add_white_background
                )
                future_to_item[future] = identifier
            
            for future in as_completed(future_to_item):
                identifier = future_to_item[future]
                try:
                    result = future.result()
                    if result:
                        results.append((identifier, result))
                        self.success_count += 1
                        print(f"BackgroundRemover.process_selected_images: Success for {identifier}, result: BytesIO")
                    else:
                        self.error_count += 1
                        print(f"BackgroundRemover.process_selected_images: Failed for {identifier}, no result")
                    if progress_callback:
                        progress_callback({
                            'processed': self.processed_count,
                            'total': total_items,
                            'successful': self.success_count,
                            'failed': self.error_count,
                            'current_item': identifier,
                            'result_item': 'BytesIO' if result else None
                        })
                except Exception as e:
                    logger.error(f"Error processing {identifier}: {e}")
                    print(f"BackgroundRemover.process_selected_images: Error for {identifier}: {e}")
                    self.error_count += 1
        
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'total_items': total_items,
            'processed': self.processed_count,
            'successful': self.success_count,
            'failed': self.error_count,
            'elapsed_time': elapsed_time,
            'avg_time_per_item': elapsed_time / max(self.processed_count, 1),
            'results': results
        }
        
        logger.info(f"Background removal completed: {self.success_count}/{total_items} successful")
        print(f"BackgroundRemover.process_selected_images: Completed, stats: {stats}")
        return stats
    
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         add_white_background: bool = False,
                         progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, any]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images (if None, creates 'no_bg' subdirectory)
            add_white_background: If True, composite onto white background
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        input_dir = Path(input_dir)
        print(f"BackgroundRemover.process_directory: Processing directory {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir / "no_bg"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        image_files = []
        
        for file_path in input_dir.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            print(f"BackgroundRemover.process_directory: No image files found in {input_dir}")
            return {
                'total_files': 0,
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'output_dir': str(output_dir)
            }
        
        logger.info(f"Found {len(image_files)} images to process")
        print(f"BackgroundRemover.process_directory: Found {len(image_files)} images")
        
        # Reset counters
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Process images with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            for image_file in image_files:
                rel_path = image_file.relative_to(input_dir)
                output_file = output_dir / f"no_bg_{rel_path.stem}.png"
                
                future = executor.submit(
                    self.remove_background,
                    Image.open(image_file).convert("RGB"),
                    add_white_background
                )
                future_to_file[future] = (image_file, output_file)
            
            for future in as_completed(future_to_file):
                image_file, output_file = future_to_file[future]
                
                try:
                    result = future.result()
                    if result:
                        # Save BytesIO to file
                        with open(output_file, 'wb') as f:
                            f.write(result.getvalue())
                        self.success_count += 1
                        print(f"BackgroundRemover.process_directory: Saved to {output_file}")
                    if progress_callback:
                        progress_callback({
                            'processed': self.processed_count,
                            'total': len(image_files),
                            'successful': self.success_count,
                            'failed': self.error_count,
                            'current_file': str(image_file),
                            'result_file': str(output_file) if result else None
                        })
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    print(f"BackgroundRemover.process_directory: Error for {image_file}: {e}")
                    self.error_count += 1
        
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'total_files': len(image_files),
            'processed': self.processed_count,
            'successful': self.success_count,
            'failed': self.error_count,
            'output_dir': str(output_dir),
            'elapsed_time': elapsed_time,
            'avg_time_per_image': elapsed_time / max(self.processed_count, 1)
        }
        
        logger.info(f"Background removal completed: {self.success_count}/{len(image_files)} successful")
        print(f"BackgroundRemover.process_directory: Completed, stats: {stats}")
        return stats
    
    def create_preview(self, original: Union[str, Path, Image.Image], 
                      processed: Union[str, Path, BytesIO]) -> Optional[Image.Image]:
        """
        Create a side-by-side preview of original and processed images.
        
        Args:
            original: Path to original image or PIL Image
            processed: Path to processed image or BytesIO
            
        Returns:
            PIL Image with side-by-side comparison or None if error
        """
        try:
            print(f"BackgroundRemover.create_preview: Creating preview")
            
            # Load original
            if isinstance(original, (str, Path)):
                original_img = Image.open(original).convert("RGB")
            else:
                original_img = original.convert("RGB")
            
            # Load processed
            if isinstance(processed, (str, Path)):
                processed_img = Image.open(processed)
            else:
                processed_img = Image.open(processed)
            
            if processed_img.mode == 'RGBA':
                white_bg = Image.new("RGB", processed_img.size, (255, 255, 255))
                mask = processed_img.split()[-1]
                white_bg.paste(processed_img, (0, 0), mask=mask)
                processed_display = white_bg
            else:
                processed_display = processed_img.convert("RGB")
            
            target_height = 300
            original_ratio = original_img.width / original_img.height
            processed_ratio = processed_display.width / processed_display.height
            
            original_resized = original_img.resize((int(target_height * original_ratio), target_height))
            processed_resized = processed_display.resize((int(target_height * processed_ratio), target_height))
            
            total_width = original_resized.width + processed_resized.width + 20
            preview = Image.new("RGB", (total_width, target_height), (240, 240, 240))
            
            preview.paste(original_resized, (0, 0))
            preview.paste(processed_resized, (original_resized.width + 20, 0))
            
            print(f"BackgroundRemover.create_preview: Preview created successfully")
            return preview
            
        except Exception as e:
            logger.error(f"Failed to create preview: {e}")
            print(f"BackgroundRemover.create_preview: Failed: {e}")
            return None


def main():
    """
    Main function for standalone usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove backgrounds from images")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("-o", "--output", help="Output directory (default: input_dir/no_bg)")
    parser.add_argument("-w", "--white-bg", action="store_true", help="Add white background")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    remover = BackgroundRemover(max_workers=args.workers)
    
    def progress_callback(progress):
        print(f"Progress: {progress['processed']}/{progress['total']} "
              f"(Success: {progress['successful']}, Failed: {progress['failed']})")
    
    stats = remover.process_directory(
        args.input_dir,
        args.output,
        args.white_bg,
        progress_callback
    )
    
    print(f"\nCompleted! {stats['successful']}/{stats['total_files']} images processed successfully")
    print(f"Output directory: {stats['output_dir']}")
    print(f"Total time: {stats['elapsed_time']:.1f}s")
    print(f"Average time per image: {stats['avg_time_per_image']:.1f}s")


if __name__ == "__main__":
    main()