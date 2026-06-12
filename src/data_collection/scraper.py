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

    In production, replace this with real scraped data.

    Args:
        product_key:      Key from SIMULATED_PRICE_RANGES.
        num_competitors:  How many competitor prices to simulate.
        seed:             Random seed for reproducibility.

    Returns:
        dict with keys: product, min_price, max_price, avg_price, competitors
    """
    import numpy as np

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
    # Demo: simulated competitor prices for rice
    data = get_competitor_price("rice_5kg")
    print(f"\nProduct  : {data['product']}")
    print(f"Avg Price: ₹{data['avg_price']}")
    print(f"Range    : ₹{data['min_price']} – ₹{data['max_price']}")
    for c in data["competitors"]:
        print(f"  {c['name']}: ₹{c['price']}")
