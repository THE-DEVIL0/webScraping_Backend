"""
eBay Image Scraper
=================

Scrapes product image URLs from eBay item or search pages.

Features:
- Concurrent product page scraping with ThreadPoolExecutor
- Extracts images from JSON-LD and CSS selectors
- Returns public image URLs (no local downloads)
- Deduplicates image URLs
- Simple logging (console with emojis or custom callback)
- Retry logic for robust requests
- Handles single item or search pages

Dependencies:
- requests
- beautifulsoup4
- concurrent.futures
- urllib3

Usage:
    from ebay_image_scraper import EbayImageScraper
    scraper = EbayImageScraper(max_workers=8)
    image_urls = scraper.scrape("https://www.ebay.com/sch/i.html?_nkw=shoes", max_products=30)
"""

import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from typing import List, Optional, Callable, Tuple
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class EbayImageScraper:
    """
    eBay image scraper that returns public image URLs.
    """

    def __init__(
        self,
        max_workers: int = 8,
        log_callback: Optional[Callable[[str, str], None]] = None,
        high_res_only: bool = True
    ):
        """
        Initialize the scraper.

        Args:
            max_workers: Max concurrent threads for scraping.
            log_callback: Optional function(msg: str, log_type: str) for logging.
            high_res_only: Filter for high-resolution images.
        """
        self.max_workers = max(1, min(max_workers, 15))
        self.log_callback = log_callback or self._default_log
        self.high_res_only = high_res_only
        self.stop_flag = False
        self.session = self._build_retry_session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _build_retry_session(self) -> requests.Session:
        """Build a requests session with retry logic."""
        sess = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
        sess.mount('https://', adapter)
        return sess

    def _default_log(self, message: str, log_type: str = "info"):
        """Default console logger with emoji."""
        emoji = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}.get(log_type, "")
        print(f"[{log_type.upper()}] {emoji} {message}")

    def log(self, message: str, log_type: str = "info"):
        """Log a message using callback or default."""
        self.log_callback(message, log_type)

    def _fetch_product_urls(self, url: str, max_products: Optional[int] = None) -> Tuple[List[str], Optional[str]]:
        """Fetch product URLs from search pages or use single item URL."""
        try:
            if '/itm/' in url:
                self.log(f"Single item URL provided: {url}", "info")
                return [url.split('?')[0]], None

            product_urls: List[str] = []
            page = 1
            while not self.stop_flag:
                page_url = f"{url}&_pgn={page}" if page > 1 else url
                self.log(f"Scanning page: {page_url}", "info")
                resp = self.session.get(page_url, verify=False, timeout=15, headers=self.headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, "html.parser")
                found_urls = [a['href'].split('?')[0] for a in soup.find_all('a', class_='s-item__link') if a.get('href') and '/itm/' in a.get('href')]
                
                if not found_urls:
                    self.log(f"No more products found on page {page}", "info")
                    break
                
                product_urls.extend(found_urls)
                self.log(f"Found {len(found_urls)} product URLs on page {page}. Total: {len(product_urls)}", "success")
                
                if max_products and len(product_urls) >= max_products:
                    self.log(f"Reached max_products limit: {max_products}", "info")
                    break
                
                page += 1
                time.sleep(0.5)  # Respectful delay

            if not product_urls:
                return [], "No product URLs found"
            
            return product_urls[:max_products], None
        except Exception as e:
            self.log(f"Error fetching product URLs: {e}", "error")
            return [], str(e)

    def _scrape_item_images(self, item_url: str) -> List[str]:
        """Scrape image URLs from a single product page."""
        try:
            self.log(f"Scraping images from {item_url}", "info")
            page = self.session.get(item_url, verify=False, timeout=20, headers=self.headers)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")
            img_urls = set()

            # JSON-LD
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.get_text(strip=True) or "{}")
                    images = []
                    if isinstance(data, dict) and "image" in data:
                        images = data["image"] if isinstance(data["image"], list) else [data["image"]]
                    for u in images:
                        if isinstance(u, str) and "ebayimg.com" in u:
                            if self.high_res_only:
                                u = u.replace("s-l64.jpg", "s-l1600.jpg").replace("s-l225.jpg", "s-l1600.jpg")
                            img_urls.add(u)
                except Exception:
                    continue

            # CSS selectors
            selectors = [
                "img#icImg",
                "img[src*='ebayimg.com']",
                "img[data-src*='ebayimg.com']",
                ".ux-image-carousel-item img",
                ".ux-image-magnifier img",
                ".image-treatment img",
                ".ux-image-gallery img",
                "div[itemprop='image'] img",
                ".vi-image-gallery__image img",
                ".img-container img",
            ]
            for sel in selectors:
                for img in soup.select(sel):
                    src = img.get("src") or img.get("data-src") or img.get("data-zoom-img")
                    if src and "ebayimg.com" in src:
                        if self.high_res_only:
                            src = src.replace("s-l64.jpg", "s-l1600.jpg").replace("s-l225.jpg", "s-l1600.jpg")
                        img_urls.add(src)

            # Filter valid extensions
            valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
            img_urls = [u for u in img_urls if any(u.lower().endswith(ext) for ext in valid_extensions)]
            
            self.log(f"Collected {len(img_urls)} image URLs from {item_url.split('/')[-1]}", "info")
            return list(img_urls)
        except Exception as e:
            self.log(f"Error scraping {item_url}: {e}", "error")
            return []

    def scrape(self, url: str, max_products: Optional[int] = 30) -> List[str]:
        """
        Scrape image URLs from eBay search or item page.

        Args:
            url: eBay search or item URL (e.g., "https://www.ebay.com/sch/i.html?_nkw=shoes").
            max_products: Max number of products to scrape (default 30).

        Returns:
            List of public image URLs.
        """
        if not url.startswith('http'):
            url = 'https://' + url

        self.url = url
        self.stop_flag = False
        self.log(f"Starting scrape for {url} with max_products={max_products}", "info")

        # Step 1: Get product URLs
        product_urls, error = self._fetch_product_urls(url, max_products)
        if error:
            self.log(f"Product discovery failed: {error}", "error")
            return []
        
        if not product_urls:
            self.log("No product URLs found", "error")
            return []

        self.log(f"Found {len(product_urls)} product pages", "success")

        # Step 2: Scrape image URLs concurrently
        self.log("Discovering image URLs...", "info")
        all_images = []

        def scrape_single(url: str) -> List[str]:
            if self.stop_flag:
                return []
            return self._scrape_item_images(url)

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
        print(f"EbayImageScraper: Returned URLs:\n{json.dumps(unique_images, indent=2)}")  # Added console log

        if not unique_images:
            self.log("No image URLs collected", "error")

        return unique_images

    def stop(self):
        """Stop ongoing scrape."""
        self.stop_flag = True
        self.log("Stop signal sent", "warning")