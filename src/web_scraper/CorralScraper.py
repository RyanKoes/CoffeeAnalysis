import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

BASE_URL = "https://www.coffeebeancorral.com"
GREEN_COFFEE_MENU_URL = f"{BASE_URL}/categories/Green-Coffee-Beans.aspx"  # main green coffee menu page


def get_subcategory_urls(menu_url):
    """Get all country/region subcategory links from the main green coffee menu."""
    r = requests.get(menu_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    subcategory_urls = []

    for a in soup.select('div.categorymenuplus a[href*="/categories/Green-Coffee-Beans/"]'):
        href = a.get('href')
        if href and href.startswith('/categories/Green-Coffee-Beans/'):
            full_url = BASE_URL + href
            if full_url not in subcategory_urls:
                subcategory_urls.append(full_url)

    return subcategory_urls


def get_product_urls(subcategory_url):
    """Get all product URLs from a subcategory page (no pagination)."""
    urls = set()  # deduplicate automatically
    r = requests.get(subcategory_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    for a in soup.find_all('a', href=re.compile(r'/product/')):
        href = a.get('href').split('#')[0]  # remove anchor fragments
        if href.startswith('/product/'):
            urls.add(BASE_URL + href)

    return list(urls)


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
    # Step 1: Get all subcategories
    subcategories = get_subcategory_urls(GREEN_COFFEE_MENU_URL)
    print(f"Found {len(subcategories)} subcategories")

    # Step 2: Get all product URLs from subcategories
    all_product_urls = []
    for subcat_url in subcategories:
        print(f"Scanning subcategory: {subcat_url}")
        products = get_product_urls(subcat_url)
        print(f"  Found {len(products)} products")
        all_product_urls.extend(products)

    # Deduplicate
    all_product_urls = list(set(all_product_urls))
    print(f"Total unique products found: {len(all_product_urls)}")

    # Step 3: Scrape each product
    all_data = []
    for i, url in enumerate(all_product_urls, 1):
        print(f"Scraping {i}/{len(all_product_urls)}: {url}")
        try:
            coffee_data = scrape_coffee_page(url)
            all_data.append(coffee_data)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        time.sleep(1)  # polite delay

    # Step 4: Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("green_coffees.csv", index=False)
    print("Saved all green coffees to green_coffees.csv")
