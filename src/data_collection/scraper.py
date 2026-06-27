"""
Competitor Price Scraper
-------------------------
Scrapes competitor product prices from public web pages using
BeautifulSoup + Requests. Returns structured data for the pricing engine.

Usage:
    from src.data_collection.scraper import scrape_prices, get_competitor_price

    prices = scrape_prices("https://example-kirana.com/products")
    avg    = get_competitor_price("rice_5kg")

NOTE:
    - Always check a site's robots.txt and Terms of Service before scraping.
    - For JS-heavy pages, switch to Selenium (selenium>=4.0, uncomment below).
    - Rate-limit your requests to avoid IP bans (REQUEST_DELAY_SECONDS).

TODO:
    - Add Selenium fallback for JS-rendered sites.
    - Add database persistence (SQLite) for price history.
    - Schedule scraping with APScheduler for real-time updates.
"""

from __future__ import annotations
import time
import logging
import re
from dataclasses import dataclass
from typing import Optional
import requests
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

REQUEST_DELAY_SECONDS = 1.5   # be polite to servers
DEFAULT_TIMEOUT       = 10    # seconds
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class ProductPrice:
    """Scraped price record for a single product."""
    product_name:  str
    price:         float
    currency:      str = "INR"
    source_url:    str = ""
    in_stock:      bool = True
    scraped_at:    str = ""

    def __post_init__(self):
        if not self.scraped_at:
            from datetime import datetime
            self.scraped_at = datetime.now().isoformat()


# ── Core Scraping Functions ───────────────────────────────────────────────────

def fetch_page(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[BeautifulSoup]:
    """
    Fetch a web page and return a BeautifulSoup object.

    Returns None on failure (network error, non-200 status, etc.)
    """
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def extract_price_from_text(text: str) -> Optional[float]:
    """
    Extract a numeric price from a string like '₹129', 'Rs. 99.50', '120.00'.

    Returns None if no valid price found.
    """
    # Strip currency symbols and commas, match decimal number
    cleaned = re.sub(r"[₹$€£Rs.\s,]", "", text.strip())
    match   = re.search(r"\d+(\.\d+)?", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def scrape_prices(
    url: str,
    product_selector: str = ".product-price",   # CSS selector for price element
    name_selector:    str = ".product-name",    # CSS selector for name element
) -> list[ProductPrice]:
    """
    Generic scraper: fetches a product listing page and extracts name + price.

    Args:
        url:               Target URL to scrape.
        product_selector:  CSS selector for price elements.
        name_selector:     CSS selector for product name elements.

    Returns:
        List of ProductPrice objects. Empty list on failure.
    """
    soup = fetch_page(url)
    if soup is None:
        return []

    prices     = soup.select(product_selector)
    names      = soup.select(name_selector)
    results    = []

    for name_el, price_el in zip(names, prices):
        name  = name_el.get_text(strip=True)
        price = extract_price_from_text(price_el.get_text())
        if price is not None:
            results.append(ProductPrice(
                product_name=name,
                price=price,
                source_url=url,
            ))

    logger.info(f"Scraped {len(results)} products from {url}")
    time.sleep(REQUEST_DELAY_SECONDS)   # polite delay
    return results


def scrape_bigbasket(
    category_slug: str = "foodgrains-oil-masala",
    max_pages: int = 1
) -> list[ProductPrice]:
    """
    Scrape BigBasket category using their embedded JSON data (__NEXT_DATA__).
    
    Args:
        category_slug: URL slug for the category, e.g. 'foodgrains-oil-masala'
        max_pages: Number of pages to scrape
        
    Returns:
        List of ProductPrice objects
    """
    results = []
    
    for page in range(1, max_pages + 1):
        url = f"https://www.bigbasket.com/cl/{category_slug}/?nc=nb&page={page}"
        logger.info(f"Scraping BigBasket: {url}")
        
        soup = fetch_page(url)
        if not soup:
            continue
            
        next_data_script = soup.find('script', id='__NEXT_DATA__')
        if not next_data_script:
            logger.warning(f"__NEXT_DATA__ not found on {url}")
            continue
            
        try:
            data = json.loads(next_data_script.string)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON")
            continue
            
        # Recursive search for 'products' list
        def find_products(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'products' and isinstance(v, list) and len(v) > 0 and 'pricing' in v[0]:
                        return v
                    res = find_products(v)
                    if res: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_products(item)
                    if res: return res
            return None

        products = find_products(data)
        if products:
            for p in products:
                desc = p.get('desc', 'Unknown')
                brand = p.get('brand', {}).get('name', 'Unknown')
                weight = p.get('w', '')
                full_name = f"{brand} {desc} {weight}".strip()
                
                pricing = p.get('pricing', {})
                discount = pricing.get('discount', {})
                prim_price = discount.get('prim_price', {})
                
                sp_str = prim_price.get('sp')
                mrp_str = discount.get('mrp')
                
                try:
                    # Prefer sale price if available, else MRP
                    price = float(sp_str) if sp_str else float(mrp_str)
                    
                    results.append(ProductPrice(
                        product_name=full_name,
                        price=price,
                        source_url=url,
                    ))
                except (ValueError, TypeError):
                    continue
                    
        time.sleep(REQUEST_DELAY_SECONDS)
        
    logger.info(f"Successfully scraped {len(results)} products from BigBasket ({category_slug})")
    return results


# ── Simulated Fallback (for demo / testing) ───────────────────────────────────

# A mapping of product categories → realistic competitor price ranges (₹)
SIMULATED_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "rice_5kg":     (220, 280),
    "dal_1kg":      (85,  130),
    "cooking_oil":  (130, 180),
    "soap_bar":     (28,   55),
    "tea_250g":     (90,  145),
    "sugar_1kg":    (42,   60),
    "atta_5kg":     (180, 240),
    "biscuits":     (20,   60),
    "chips":        (10,   40),
    "cold_drink":   (15,   60),
}


def get_competitor_price(
    product_key: str,
    num_competitors: int = 3,
    seed: int | None = None,
) -> dict:
    """
    Return simulated competitor price data for a product.
    If real scraped data is available in data/raw/bb_competitor_prices.json, it will be used instead.

    Args:
        product_key:      Key from SIMULATED_PRICE_RANGES or category name.
        num_competitors:  How many competitor prices to return.
        seed:             Random seed for reproducibility.

    Returns:
        dict with keys: product, min_price, max_price, avg_price, competitors
    """
    import numpy as np
    from pathlib import Path
    
    # Try to load real scraped data first
    bb_data_path = Path(__file__).parents[2] / "data" / "raw" / "bb_competitor_prices.json"
    if bb_data_path.exists():
        try:
            with open(bb_data_path, "r") as f:
                bb_prices = json.load(f)
            
            if bb_prices:
                # Randomly sample from real prices to provide variety
                rng = np.random.default_rng(seed)
                sample = rng.choice(bb_prices, size=min(num_competitors, len(bb_prices)), replace=False)
                
                prices = [p["price"] for p in sample]
                
                return {
                    "product": f"Real Data ({len(bb_prices)} scraped)",
                    "min_price": round(min(prices), 2),
                    "max_price": round(max(prices), 2),
                    "avg_price": round(sum(prices) / len(prices), 2),
                    "competitors": [
                        {"name": p["product_name"][:30] + "..." if len(p["product_name"]) > 30 else p["product_name"], 
                         "price": p["price"]}
                        for p in sample
                    ],
                }
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}. Falling back to simulation.")

    if product_key not in SIMULATED_PRICE_RANGES:
        product_key = "rice_5kg"   # default fallback

    lo, hi = SIMULATED_PRICE_RANGES[product_key]
    rng    = np.random.default_rng(seed)
    competitor_prices = rng.uniform(lo, hi, num_competitors).round(2).tolist()

    return {
        "product":      product_key,
        "min_price":    round(min(competitor_prices), 2),
        "max_price":    round(max(competitor_prices), 2),
        "avg_price":    round(sum(competitor_prices) / len(competitor_prices), 2),
        "competitors":  [
            {"name": f"Competitor {chr(65+i)}", "price": p}
            for i, p in enumerate(competitor_prices)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Selenium upgrade (uncomment for JS-rendered pages):
#
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
#
# def fetch_js_page(url: str) -> Optional[BeautifulSoup]:
#     opts = Options()
#     opts.add_argument("--headless")
#     driver = webdriver.Chrome(options=opts)
#     driver.get(url)
#     soup = BeautifulSoup(driver.page_source, "html.parser")
#     driver.quit()
#     return soup
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Demo: Scrape real data from BigBasket
    print("Scraping real competitor prices from BigBasket...")
    bb_prices = scrape_bigbasket("foodgrains-oil-masala", max_pages=1)
    
    if bb_prices:
        print(f"\nFound {len(bb_prices)} products.")
        print("\nTop 5 products:")
        for p in bb_prices[:5]:
            print(f"  - {p.product_name}: Rs. {p.price}")
            
        # Save to JSON for analysis/dashboard
        from pathlib import Path
        output_path = Path(__file__).parents[2] / "data" / "raw" / "bb_competitor_prices.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump([p.__dict__ for p in bb_prices], f, indent=2)
        print(f"\nSaved scraped data to {output_path}")
    else:
        print("Failed to scrape BigBasket.")
