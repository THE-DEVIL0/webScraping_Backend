"""
Shopify Image Scraper
====================

Scrapes product image URLs from Shopify store product pages.

Features:
- Concurrent product page scraping with ThreadPoolExecutor
- Extracts images from JSON-LD, og:image, and img tags
- Returns public image URLs (no local downloads)
- Deduplicates image URLs
- Simple logging (console or custom callback)
- No MongoDB storage

Dependencies:
- requests
- beautifulsoup4
- concurrent.futures

Usage:
    from shopify_image_scraper import ShopifyImageScraper
    scraper = ShopifyImageScraper(max_workers=8)
    image_urls = scraper.scrape("https://example.com", max_products=30)
"""

import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from typing import List, Optional, Callable
import json
import time

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ShopifyImageScraper:
    """
    Shopify image scraper that returns public image URLs.
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the scraper.

        Args:
            max_workers: Max concurrent threads for scraping.
            log_callback: Optional function(msg: str, log_type: str) for logging.
        """
        self.max_workers = min(max_workers, 15)
        self.log_callback = log_callback or self._default_log
        self.stop_flag = False

    def _default_log(self, message: str, log_type: str = "info"):
        """Default console logger with emoji."""
        emoji = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}.get(log_type, "")
        print(f"[{log_type.upper()}] {emoji} {message}")

    def log(self, message: str, log_type: str = "info"):
        """Log a message using callback or default."""
        self.log_callback(message, log_type)

    def _fetch_product_urls(self, store_url: str, max_products: Optional[int] = None) -> tuple[List[str], Optional[str]]:
        """Fetch product URLs from Shopify sitemap or products page."""
        try:
            parsed_url = requests.utils.urlparse(store_url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            # Try multiple sitemap URLs
            sitemap_urls = [
                f"{domain}/sitemap_products_1.xml",  # Standard sitemap
                f"{domain.replace('www.', '')}/sitemap_products_1.xml",  # Without www
                f"{domain}/sitemap.xml"  # Main sitemap
            ]
            
            product_urls = []
            error = None
            
            for sitemap_url in sitemap_urls:
                try:
                    self.log(f"Trying sitemap: {sitemap_url}", "info")
                    resp = requests.get(sitemap_url, verify=False, timeout=10)
                    resp.raise_for_status()
                    
                    soup = BeautifulSoup(resp.content, "xml")
                    product_urls = [loc.text for loc in soup.find_all("loc") if "/products/" in loc.text]
                    
                    if product_urls:
                        self.log(f"Found {len(product_urls)} product URLs in sitemap", "success")
                        if max_products:
                            product_urls = product_urls[:max_products]
                        return product_urls, None
                    else:
                        self.log(f"No product URLs found in {sitemap_url}", "warning")
                except Exception as e:
                    self.log(f"Sitemap {sitemap_url} failed: {e}", "error")
                    error = str(e)
            
            # Fallback: Scrape /products or /collections/all page
            self.log("Falling back to /products or /collections/all page scraping", "warning")
            for fallback_url in [f"{domain}/products", f"{domain}/collections/all"]:
                try:
                    resp = requests.get(fallback_url, verify=False, timeout=10)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.content, "html.parser")
                    
                    # Find product links (common Shopify selectors)
                    product_urls = []
                    for a in soup.find_all("a", href=True):
                        href = a['href']
                        if "/products/" in href:
                            full_url = href if href.startswith('http') else f"{domain}{href}"
                            product_urls.append(full_url)
                    
                    # Deduplicate and limit
                    product_urls = list(set(product_urls))
                    if max_products:
                        product_urls = product_urls[:max_products]
                    
                    if product_urls:
                        self.log(f"Found {len(product_urls)} product URLs on {fallback_url}", "success")
                        return product_urls, None
                    else:
                        self.log(f"No product URLs found on {fallback_url}", "warning")
                except Exception as e:
                    self.log(f"{fallback_url} scraping failed: {e}", "error")
                    error = str(e)
            
            return [], "No product URLs found in sitemap or fallback pages"
        
        except Exception as e:
            self.log(f"Product discovery failed: {e}", "error")
            return [], str(e)

    def _scrape_product_images(self, url: str) -> List[str]:
        """Scrape image URLs from a single product page."""
        try:
            page = requests.get(url, verify=False, timeout=10)
            page.raise_for_status()
            
            soup = BeautifulSoup(page.content, "html.parser")
            img_urls = set()

            # JSON-LD structured data
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and "image" in data:
                        images = data["image"] if isinstance(data["image"], list) else [data["image"]]
                        img_urls.update(images)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "image" in item:
                                images = item["image"] if isinstance(item["image"], list) else [item["image"]]
                                img_urls.update(images)
                except:
                    continue

            # Fallback: og:image and img tags
            if not img_urls:
                og_img = soup.find('meta', property='og:image')
                if og_img and og_img.get('content'):
                    img_urls.add(og_img['content'])
                
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src')
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        if src.startswith('//'):
                            src = 'https:' + src
                        img_urls.add(src)

            self.log(f"Collected {len(img_urls)} image URLs from {url.split('/')[-1]}", "info")
            return list(img_urls)
        except Exception as e:
            self.log(f"Error scraping {url}: {e}", "error")
            return []

    def scrape(self, store_url: str, max_products: int = 30) -> List[str]:
        """
        Scrape image URLs from Shopify store.

        Args:
            store_url: Shopify store URL (e.g., "https://allbirds.com").
            max_products: Max number of products to scrape (default 30).

        Returns:
            List of public image URLs.
        """
        if not store_url.startswith('http'):
            store_url = 'https://' + store_url

        self.stop_flag = False
        self.log("Starting product discovery...", "info")

        # Step 1: Get product URLs
        product_urls, error = self._fetch_product_urls(store_url, max_products)
        if error:
            self.log(f"Product discovery failed: {error}", "error")
            return []
        
        if not product_urls:
            self.log("No products found in sitemap or fallback pages", "error")
            return []

        self.log(f"Found {len(product_urls)} product pages", "success")

        # Step 2: Scrape image URLs concurrently
        self.log("Discovering image URLs...", "info")
        all_images = []

        def scrape_single(url: str) -> List[str]:
            if self.stop_flag:
                return []
            images = self._scrape_product_images(url)
            if images:
                self.log(f"Collected {len(images)} image URLs from {url.split('/')[-1]}", "info")
            return images

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(scrape_single, url): url for url in product_urls}
            
            for future in as_completed(future_to_url):
                if self.stop_flag:
                    break
                try:
                    images = future.result()
                    all_images.extend(images)
                except Exception as e:
                    self.log(f"Error scraping {future_to_url[future]}: {e}", "error")

        # Deduplicate
        unique_images = list(set(all_images))
        self.log(f"Collected {len(unique_images)} unique image URLs (removed {len(all_images) - len(unique_images)} duplicates)", "success")
        print(f"ShopifyImageScraper: Returned URLs:\n{json.dumps(unique_images, indent=2)}")  # Added console log

        if not unique_images:
            self.log("No image URLs collected", "error")

        return unique_images

    def stop(self):
        """Stop ongoing scrape."""
        self.stop_flag = True
        self.log("Stop signal sent", "warning")