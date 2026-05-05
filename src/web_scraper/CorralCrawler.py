import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

BASE_URL = "https://www.coffeebeancorral.com"
ALL_COFFEES_URL = f"{BASE_URL}/categories/Green-Coffee-Beans/All-Coffees.aspx?q=&o=1&p={{page}}&i=12&d=12"

def get_all_product_urls():
    """Loop through all paginated pages and collect product URLs."""
    page = 1
    urls = []

    while True:
        url = ALL_COFFEES_URL.format(page=page)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Grab all product links in the current page
        product_divs = soup.select('.SingleProductDisplayName a')
        if not product_divs:
            break  # no more products, stop the loop

        for a in product_divs:
            href = a['href'].split('#')[0]  # remove anchors
            full_url = BASE_URL + href
            if full_url not in urls:
                urls.append(full_url)

        print(f"Page {page}: found {len(product_divs)} products")
        page += 1
        time.sleep(1)  # polite delay

    return urls

def scrape_coffee_page(url):
    """Scrape a single Coffee Bean Corral product page."""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    data = {'URL': url}

    # --- Name ---
    name_tag = soup.find('h1')
    if name_tag:
        data['Name'] = " ".join(name_tag.text.split())

    # --- Cupping Notes ---
    cupping_label = soup.find('div', string=re.compile('Cupping Notes', re.I))
    if cupping_label:
        note_span = cupping_label.find_next('span', id=re.compile('lblShortDescription'))
        if note_span:
            data['Cupping Notes'] = note_span.get_text(strip=True)

    # --- Attributes (1–7 scale) ---
    for div in soup.select('.spd-matrix-attributes div.d-flex'):
        label_tag = div.find('label')
        span_tag = div.find('span', class_=re.compile('spd-matrix-attribute'))
        if label_tag and span_tag:
            label = label_tag.get_text(strip=True)
            match = re.search(r'attribute-(\d+)', span_tag['class'][0])
            if match:
                data[label] = int(match.group(1))

    # --- Flavors (1–4 scale) ---
    for div in soup.select('.spd-matrix-flavors div.d-flex'):
        label_tag = div.find('label')
        span_tag = div.find('span', class_=re.compile('spd-matrix-flavor'))
        if label_tag and span_tag:
            label = label_tag.get_text(strip=True)
            match = re.search(r'flavor-(\d+)', span_tag['class'][0])
            if match:
                data[label] = int(match.group(1))

    # --- Specifications (Category, Country, etc.) ---
    specs_ul = soup.select_one('.producttypepanel ul.typedisplay')
    if specs_ul:
        for li in specs_ul.find_all('li'):
            label_tag = li.find('span', class_='productpropertylabel')
            value_tag = li.find('span', class_='productpropertyvalue')
            if label_tag and value_tag:
                data[label_tag.get_text(strip=True)] = value_tag.get_text(strip=True)

    return data

if __name__ == "__main__":
    # Step 1: Get all product URLs
    print("Collecting all product URLs...")
    product_urls = get_all_product_urls()
    print(f"Total products found: {len(product_urls)}")

    # Step 2: Scrape each product
    all_data = []
    for i, url in enumerate(product_urls, 1):
        print(f"Scraping {i}/{len(product_urls)}: {url}")
        try:
            coffee_data = scrape_coffee_page(url)
            all_data.append(coffee_data)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        time.sleep(1)  # polite delay

    # Step 3: Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("corral_green_coffees.csv", index=False)
    print("Saved all green coffees to green_coffees.csv")
