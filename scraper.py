from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import os


def scrape_website(page_number):
    """Scrape a single page and return a list of row dicts."""
    url = f'https://www.politifact.com/factchecks/list/?page={page_number}'
    try:
        webpage = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        webpage.raise_for_status()
    except requests.RequestException as e:
        print(f"[Page {page_number}] Request failed: {e}")
        return []

    soup = BeautifulSoup(webpage.text, 'html.parser')

    statement_footer = soup.find_all('footer', class_='m-statement__footer')
    statement_quote  = soup.find_all('div',    class_='m-statement__quote')
    statement_meta   = soup.find_all('div',    class_='m-statement__meta')
    statement_meter  = soup.find_all('div',    class_='m-statement__meter')

    # Parse each field into a flat list first
    parsed_authors, parsed_dates = [], []
    for f in statement_footer:
        text = f.text.strip().split()
        if len(text) >= 7:
            parsed_authors.append(text[1] + ' ' + text[2])
            parsed_dates.append(text[4] + ' ' + text[5] + ' ' + text[6])
        else:
            parsed_authors.append(None)
            parsed_dates.append(None)

    parsed_statements = []
    for q in statement_quote:
        a = q.find('a')
        parsed_statements.append(a.text.strip() if a else None)

    parsed_sources = []
    for m in statement_meta:
        a = m.find('a')
        parsed_sources.append(a.text.strip() if a else None)

    parsed_targets = []
    for t in statement_meter:
        img = t.find('img')
        parsed_targets.append(img.get('alt', '').strip() if img else None)

    # Zip into rows — stops at the shortest list so nothing misaligns
    rows = []
    for author, date, statement, source, target in zip(
        parsed_authors, parsed_dates, parsed_statements, parsed_sources, parsed_targets
    ):
        rows.append({
            'author':    author,
            'date':      date,
            'statement': statement,
            'source':    source,
            'verdict':   target,
        })

    return rows


def run_scraper(n_pages=300, output_path="data/politifact_data.csv"):
    all_rows = []

    for i in range(1, n_pages + 1):
        print(f"Scraping page {i}/{n_pages}...")
        rows = scrape_website(i)
        all_rows.extend(rows)
        time.sleep(0.5)

    data = pd.DataFrame(all_rows)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\nSaved {len(data)} rows to {output_path}")
    return data


if __name__ == "__main__":
    run_scraper()