#!/usr/bin/env python3
"""
Optimized Amazon Image Scraper (2025 Edition)
Fast extraction of 100+ image URLs in seconds from Amazon stores/searches.
Uses async HTTP requests + BeautifulSoup for speed (no Selenium).
Renamed to AmazonImageScraper for compatibility with services.py.

Author: Grok (xAI)
Version: 2.0.5 - Returns URLs instead of downloading, enhanced selectors
Requirements: pip install aiohttp beautifulsoup4
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import logging
import random
import json
import time
from aiohttp import ClientResponseError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmazonImageScraper:
    def __init__(self, url: str, max_workers: int = 8, log_callback: Optional[callable] = None, headless: bool = True, **kwargs):
        """
        Initialize the fast Amazon scraper, compatible with original interface.
        
        Args:
            url: Amazon store or search URL (e.g., https://www.amazon.com/s?k=electronics)
            max_workers: Not used (kept for compatibility), uses internal concurrency
            log_callback: Optional logging callback for services.py
            headless: Ignored (no Selenium), kept for compatibility
            **kwargs: Absorb extra args (e.g., driver_type) for compatibility
        """
        self.url = url
        self.log_callback = log_callback or (lambda msg, log_type: logger.log(logging.INFO if log_type in ["info", "success"] else logging.ERROR, msg))
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Headers for anti-detection
        self.headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
        ]
        
        # Pagination pattern
        self.is_store = '/stores/' in url
        if self.is_store:
            base_url = url.split('?')[0]
            self.page_pattern = base_url.replace('/page/', '/page/{}/') + (('?' + url.split('?')[1]) if '?' in url else '')
        else:
            self.page_pattern = url + ('' if '?' in url else '?') + '&page={}'
        
        self.log_callback(f"Initialized AmazonImageScraper for {url}", "info")

    async def create_session(self):
        """Create aiohttp session with high concurrency."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=random.choice(self.headers_list)
        )

    async def fetch_page(self, page_num: int, max_retries: int = 3, backoff_factor: float = 2.0) -> Optional[str]:
        """Fetch a single page asynchronously with retries for 429 and 503 errors."""
        url = self.page_pattern.format(page_num) if page_num > 1 else self.url
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        self.log_callback(f"Fetched page {page_num}: {len(html)} chars", "info")
                        return html
                    elif response.status in (429, 503):
                        self.log_callback(f"Rate limited on page {page_num}: HTTP {response.status}, attempt {attempt + 1}/{max_retries}", "error")
                        if attempt < max_retries - 1:
                            delay = backoff_factor ** attempt
                            self.log_callback(f"Retrying after {delay}s", "info")
                            await asyncio.sleep(delay)
                        continue
                    else:
                        self.log_callback(f"Failed to fetch page {page_num}: HTTP {response.status}", "error")
                        return None
            except ClientResponseError as e:
                if e.status in (429, 503):
                    self.log_callback(f"Rate limited on page {page_num}: HTTP {e.status}, attempt {attempt + 1}/{max_retries}", "error")
                    if attempt < max_retries - 1:
                        delay = backoff_factor ** attempt
                        self.log_callback(f"Retrying after {delay}s", "info")
                        await asyncio.sleep(delay)
                        continue
                self.log_callback(f"Error fetching page {page_num}: {e}", "error")
                return None
            except Exception as e:
                self.log_callback(f"Error fetching page {page_num}: {e}", "error")
                return None
        self.log_callback(f"Failed to fetch page {page_num} after {max_retries} attempts", "error")
        return None

    def parse_product_images(self, html: str) -> List[Dict[str, str]]:
        """Parse HTML for product data: name, ASIN, image URLs."""
        soup = BeautifulSoup(html, 'html.parser')
        products = []
        
        # 2025 selectors, expanded for Razer and other store pages
        containers = soup.select(
            'div[data-component-type="s-search-result"], '
            '.s-result-item[data-asin], '
            'div[data-asin], '
            '.s-product-image-container, '
            '.a-section.a-spacing-base, '
            '.sg-col-inner > div > div > a, '
            '.storefront-product, '
            'div[data-testid="storefront-product"], '
            '.product-grid a, '
            '.grid-item-container, '
            '.storefront-product-card, '
            '.product-card, '
            '.grid-item, '
            'div[class*="product"], '
            'a[class*="product"]'
        )
        
        if not containers:
            self.log_callback("No product containers found, check selectors or page structure", "error")
        
        for container in containers:
            try:
                # Log container for debugging
                self.log_callback(f"Processing container: {str(container)[:100]}...", "info")
                
                # ASIN
                link_elem = container.select_one('a[href*="/dp/"], a[href*="/gp/product/"]')
                asin = container.get('data-asin', 'unknown')
                if link_elem:
                    href = link_elem.get('href', '')
                    if '/dp/' in href:
                        asin = href.split('/dp/')[1].split('?')[0].split('/')[0]
                
                # Name
                name_elem = container.select_one(
                    'h2 span, .a-size-base-plus, .a-text-normal, '
                    'span.a-size-medium, .product-title, '
                    'h3, h4, .product-title-text, span'
                )
                name = name_elem.get_text(strip=True) if name_elem else f"Product_{asin}"
                name = ''.join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()[:50].replace(' ', '_')
                
                # Image
                img_elem = container.select_one(
                    'img[data-a-dynamic-image], img.s-image, '
                    'img.product-image, img[class*="product"], img'
                )
                if not img_elem:
                    self.log_callback(f"No image found in container: {str(container)[:100]}...", "error")
                    continue
                
                dynamic_data = img_elem.get('data-a-dynamic-image', '{}')
                try:
                    img_dict = json.loads(dynamic_data)
                    hi_res_url = max(img_dict.keys(), key=len) if img_dict else img_elem.get('src', '')
                except json.JSONDecodeError:
                    hi_res_url = img_elem.get('src', '')
                
                if hi_res_url and ('.jpg' in hi_res_url or '.png' in hi_res_url):
                    if hi_res_url.startswith('//'):
                        hi_res_url = 'https:' + hi_res_url
                    elif hi_res_url.startswith('/'):
                        hi_res_url = 'https://www.amazon.com' + hi_res_url
                    products.append({
                        'name': name,
                        'asin': asin,
                        'url': hi_res_url
                    })
                    self.log_callback(f"Added product: {name} (ASIN: {asin})", "info")
                else:
                    self.log_callback(f"Invalid image URL: {hi_res_url}", "error")
            except Exception as e:
                self.log_callback(f"Parse error in container: {str(container)[:100]}...: {e}", "error")
                continue
        
        self.log_callback(f"Parsed {len(products)} products", "info")
        return products

    async def _fetch_product_urls(self, max_products: Optional[int], *args, **kwargs) -> List[str]:
        """Fetch product URLs for services.py compatibility."""
        products = await self.collect_all_images(max_products or 100)
        return [f"https://www.amazon.com/dp/{p['asin']}" for p in products]

    async def collect_all_images(self, target_images: int) -> List[Dict[str, str]]:
        """Collect image data from pages concurrently."""
        await self.create_session()
        num_pages = min((target_images // 20) + 1, 5)
        tasks = [self.fetch_page(i) for i in range(1, num_pages + 1)]
        pages_html = await asyncio.gather(*tasks)
        
        all_products = []
        for i, html in enumerate(pages_html, 1):
            if html:
                products = self.parse_product_images(html)
                all_products.extend(products)
                if len(all_products) >= target_images * 2:
                    break
        
        seen_urls = set()
        unique_products = [p for p in all_products if p['url'] not in seen_urls and not seen_urls.add(p['url'])]
        
        self.log_callback(f"Collected {len(unique_products)} unique image URLs", "success")
        await self.session.close()
        return unique_products[:target_images]

    async def scrape_async(self, max_products: Optional[int] = None) -> List[str]:
        """Async scrape method returning public image URLs."""
        target_images = max_products * 5 if max_products else 100
        products = await self.collect_all_images(target_images)
        image_urls = [p['url'] for p in products]
        print(f"AmazonImageScraper: Returned URLs:\n{json.dumps(image_urls, indent=2)}")  # Added console log
        self.log_callback(f"Returning {len(image_urls)} image URLs", "success")
        return image_urls

    def scrape(self, max_products: Optional[int] = None, headless: bool = True, driver_type: str = "edge") -> List[str]:
        """Sync wrapper for services.py compatibility."""
        return asyncio.run(self.scrape_async(max_products))