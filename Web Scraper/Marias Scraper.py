import requests
from bs4 import BeautifulSoup
import csv
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from collections import defaultdict
import json

# --- Configuration ---
BASE = "https://www.sweetmarias.com"
CATALOG_URL = f"{BASE}/green-coffee.html"
SLEEP = 2  # seconds between requests
TIMEOUT = 20


# --- Utility Functions (Selenium/Browser) ---

def fetch(url):
    """Fetches a URL using Selenium (Chrome VISIBLE) for manual bypass."""
    options = Options()

    # 🚨 CRITICAL: REMOVE the headless argument 🚨
    # options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("start-maximized")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        print("\n=======================================================")
        print("🚨 ACTION REQUIRED: MANUALLY SOLVE THE CAPTCHA 🚨")
        print("A browser window is open. Please check the 'I am human' box.")
        print("The script will wait 60 seconds for you to solve it.")
        print("=======================================================\n")

        # Give yourself time to manually solve the Turnstile challenge
        time.sleep(60)

        html = driver.page_source

        # Debug to confirm success (should see product names now)
        print(f"--- DEBUG: Page Source Length: {len(html)} bytes ---")

    finally:
        driver.quit()
    return html


def slugify(text):
    """Converts a coffee name to a URL-friendly slug."""
    text = text.lower().replace(' ', '-')
    # Remove non-word characters (except for the hyphen)
    text = re.sub(r'[^\w\-]', '', text)
    # Replace multiple hyphens with a single hyphen
    text = re.sub(r'\-\-+', '-', text)
    return text.strip('-')


# --- Part 1: Crawl and Collect Product URLs ---

def extract_products_from_html(html_content):
    """Extracts product data (name/price) from the JavaScript 'dataLayer' variable."""
    products = []

    # Regex to capture the JSON array assigned to 'var dl4Objects = [...]'
    match = re.search(r'var dl4Objects = (\[.*?\]);', html_content, re.DOTALL)

    if match:
        json_data_str = match.group(1).strip().rstrip(';')

        try:
            data = json.loads(json_data_str)

            for obj in data:
                if 'ecommerce' in obj and 'items' in obj['ecommerce']:
                    for item in obj['ecommerce']['items']:
                        name = item.get('item_name')
                        if name:
                            # Construct the URL based on the known slug format
                            slug = slugify(name)
                            url = f"{BASE}/{slug}.html"
                            products.append({"url": url, "title": name})

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from dataLayer: {e}")

    return products


def get_all_product_urls():
    """Crawl catalog pages and collect all product URLs and initial names."""
    products_map = {}  # Use a map to store unique URLs and names
    next_page = CATALOG_URL
    page_num = 1
    max_pages = 10  # Safety limit

    while next_page:
        print(f"Fetching catalog page {page_num}: {next_page}")

        try:
            html = fetch(next_page)
            # Extract data from the dataLayer in the HTML source
            new_products = extract_products_from_html(html)

            if not new_products and page_num > 1:
                # If we found items on previous pages but none now, assume end of catalog
                break

            for p in new_products:
                products_map[p['url']] = p['title']

            # Find next page link using BeautifulSoup, looking for the <link rel="next"> tag
            soup = BeautifulSoup(html, "html.parser")
            nxt_link = soup.find("link", rel="next")
            next_page = nxt_link["href"] if nxt_link else None

            if page_num >= max_pages:
                print(f"Stopping at max page limit ({max_pages}).")
                next_page = None

            page_num += 1
            time.sleep(SLEEP)

        except Exception as e:
            print(f"Error during catalog crawling on page {page_num}: {e}. Stopping.")
            break

    print(f"Found {len(products_map)} product URLs.")
    return list(products_map.keys())


# --- Part 2: Extract Chart Data from Product Page ---

def extract_chart_data(soup, chart_selector_key, data_prefix):
    """
    Finds a chart container and extracts key-value pairs based on dt/dd elements.
    """
    data = {}

    # This selector targets the main detail block for Grading and Flavor
    chart = soup.select_one(f'div.product.attribute.{chart_selector_key} div.value')
    if not chart:
        return data

    # Assumes the data is structured as a definition list (dt=key, dd=value)
    dt_elements = chart.select('dt')
    for dt in dt_elements:
        key = dt.get_text(strip=True).replace(':', '').strip()
        dd = dt.find_next_sibling('dd')
        if dd:
            value = dd.get_text(strip=True)
            if key and value:
                # Prefix the key to distinguish between Scoring and Notes fields
                data[f"{data_prefix}_{key}"] = value

    return data


def parse_product_page(url):
    """Extract all grading and tasting notes from a single product page."""
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    # Initialize data dictionary, using defaultdict for safe key access
    data = defaultdict(lambda: None, {"url": url})

    # 1. Title (Coffee Name)
    h1 = soup.find("h1", {"class": "page-title"})
    if h1:
        data["title"] = h1.get_text(strip=True)

    # 2. Extract 'Scoring' Chart Data (Class: 'grading')
    # CSS selector targets the grading block based on its parent class
    scoring_data = extract_chart_data(soup, 'grading', 'Scoring')
    data.update(scoring_data)

    # 3. Extract 'Tasting Notes' Chart Data (Class: 'flavor')
    # CSS selector targets the flavor notes block
    tasting_data = extract_chart_data(soup, 'flavor', 'Notes')
    data.update(tasting_data)

    return dict(data)


# --- Part 3: Main Execution ---

def main():
    product_urls = get_all_product_urls()
    results = []
    total = len(product_urls)

    # The set of all field names found across all coffees
    all_fieldnames = {"url", "title"}

    for i, url in enumerate(product_urls, start=1):
        try:
            print(f"[{i}/{total}] Scraping {url}")
            data = parse_product_page(url)
            results.append(data)

            # Dynamically collect all keys for the CSV header
            all_fieldnames.update(data.keys())

        except Exception as e:
            print(f"Error parsing {url}: {e}")
        time.sleep(SLEEP)

    # Prepare fieldnames for CSV: ensure fixed fields come first, then others sorted alphabetically
    fixed_fields = ["url", "title"]
    # Filter out fixed fields and sort the rest
    other_fields = sorted(list(all_fieldnames - set(fixed_fields)))
    final_fieldnames = fixed_fields + other_fields

    # Save results
    csv_filename = "sweetmarias_coffees.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        # Use the dynamically created list of all fields
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done! Saved {len(results)} coffees to {csv_filename} with {len(final_fieldnames)} columns.")


if __name__ == "__main__":
    main()