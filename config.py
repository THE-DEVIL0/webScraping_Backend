# Configuration file for Amazon Image Scraper
# Modify these settings according to your needs
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Amazon Store Configuration
AMAZON_STORE_URL = "https://www.amazon.com/stores/Jabra/page/84E991A6-9AC8-4C5D-923C-2C9075D04F62?is_byline_deeplink=true&deeplink=85E9D861-B579-4D2E-9FCF-859EB5965F05&redirect_store_id=84E991A6-9AC8-4C5D-923C-2C9075D04F62&lp_asin=B0C1DBQQS5&ref_=ast_bln&store_ref=bl_ast_dp_brandLogo_sto"
# Scraping Limits
MAX_PRODUCTS = 20  # Maximum number of products to scrape
MAX_RETRIES = 3    # Maximum retry attempts for failed operations

# Browser Configuration
HEADLESS = False          # Set to False for debugging (shows browser window)
DRIVER_TYPE = "edge"      # Driver to use: "auto", "chrome", or "edge" (Edge recommended for speed and efficiency)
EDGE_DRIVER_PATH = "edgedriver_win64/msedgedriver.exe"  # Path to Edge WebDriver
CHROME_DRIVER_PATH = "chromedriver.exe"  # Path to Chrome WebDriver
WINDOW_WIDTH = 1920       # Browser window width
WINDOW_HEIGHT = 1080      # Browser window height

# Timing and Delays (in seconds)
DOWNLOAD_DELAY = 0.8      # Delay between image downloads
SCROLL_DELAY = 1.0        # Delay between scroll operations
PAGE_LOAD_TIMEOUT = 60    # Page load timeout (optimized)
ELEMENT_WAIT_TIMEOUT = 30  # Element wait timeout (optimized)
PRODUCT_DELAY = 2.0       # Delay between processing products (optimized)

# Performance Settings
MAX_WORKERS = 10           # Maximum concurrent download workers
BATCH_SIZE = 25           # Number of products to process in each batch (None for auto)
ENABLE_BATCH_PROCESSING = True  # Enable batch processing for large stores
MEMORY_THRESHOLD_MB = 1000  # Memory threshold for driver restart (increased for less frequent restarts)
PRODUCTS_BEFORE_RESTART = 20  # Number of products to process before restart (increased)

# Anti-Detection Settings
ENABLE_USER_AGENT_ROTATION = True
ENABLE_WINDOW_SIZE_RANDOMIZATION = True
ENABLE_DELAY_RANDOMIZATION = True
LIGHT_STEALTH_MODE = True   # Use light stealth for slow connections

# User Agents for Rotation (add more if needed)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

# HTTP Headers
HTTP_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Output Configuration
OUTPUT_DIRECTORY = "images"           # Directory to save downloaded images
LOG_FILE = "amazon_scraper.log"       # Log file name
LOG_LEVEL = "INFO"                    # Logging level (DEBUG, INFO, WARNING, ERROR)

# Image Processing
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
DEFAULT_EXTENSION = '.jpg'            # Default extension if none detected
MAX_FILENAME_LENGTH = 100             # Maximum length for product names

# CSS Selectors for Product Links
PRODUCT_LINK_SELECTORS = [
    "a[href*='/dp/']",
    "a[href*='/gp/product/']",
    "a[data-component-type='s-product-image']",
    ".s-result-item a[href*='/dp/']",
    ".s-result-item a[href*='/gp/product/']"
]

# CSS Selectors for Product Images
PRODUCT_IMAGE_SELECTORS = [
    "#landingImage",
    "#imgBlkFront",
    ".a-dynamic-image",
    "#main-image",
    ".a-image-container img",
    "[data-old-hires]",
    ".a-button-thumbnail img",
    "#altImages .item img"
]

# CSS Selectors for Product Names
PRODUCT_NAME_SELECTORS = [
    "#productTitle",
    ".a-size-large.product-title-word-break",
    "h1.a-size-large",
    ".a-size-large.a-color-base"
]

# Browser Options (additional arguments)
EDGE_OPTIONS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-extensions",
    "--disable-plugins",
    "--disable-web-security",
    "--allow-running-insecure-content",
    "--disable-features=VizDisplayCompositor"
]

CHROME_OPTIONS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-extensions",
    "--disable-web-security",
    "--disable-features=VizDisplayCompositor",
    "--memory-pressure-off",
    "--max_old_space_size=4096"
]

# Proxy Configuration (optional)
USE_PROXY = False
PROXY_LIST = [
    # Add your proxy servers here if needed
    # "http://proxy1:port",
    # "http://proxy2:port"
]

# Rate Limiting
ENABLE_RATE_LIMITING = True
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds

# Error Handling
CONTINUE_ON_ERROR = True      # Continue scraping even if some products fail
SKIP_FAILED_PRODUCTS = True   # Skip products that fail to load
LOG_FAILED_URLS = True        # Log URLs that failed to process

# Debug Settings
VERBOSE_LOGGING = False       # Enable verbose logging for debugging
SAVE_FAILED_PAGES = False     # Save HTML of failed pages for debugging
SHOW_BROWSER_CONSOLE = False  # Show browser console output

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")