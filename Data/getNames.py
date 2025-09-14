# ------------------------
# Requirements:
# pip install requests tqdm
# ------------------------

import requests
import csv
from tqdm import tqdm
import time
import random

# ------------------------
# Crossref API helpers
# ------------------------
doi_cache = {}

def fetch_doi_data(doi):
    if doi in doi_cache:
        return doi_cache[doi]

    url = f"https://api.crossref.org/works/{doi}"
    headers = {'User-Agent': 'DOIInfoFetcher/1.0 (mailto:your-email@example.com)'}
    try:
        time.sleep(random.uniform(0.1, 0.3))  # small jitter
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()['message']
            doi_cache[doi] = data
            return data
        elif r.status_code == 404:
            print(f"DOI {doi} not found")
        else:
            print(f"Error fetching {doi}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Exception fetching {doi}: {str(e)}")
    doi_cache[doi] = None
    return None

def parse_doi_data(doi):
    data = fetch_doi_data(doi)
    if data is None:
        return {
            "DOI": doi,
            "Title": "",
            "Authors": "",
            "Year": ""
        }

    # Title
    title = data.get("title", [""])[0]

    # Authors
    authors_list = data.get("author", [])
    authors = []
    for a in authors_list:
        given = a.get("given", "")
        family = a.get("family", "")
        full_name = f"{given} {family}".strip()
        if full_name:
            authors.append(full_name)
    authors_str = "; ".join(authors)

    # Year
    year = ""
    if "published-print" in data and "date-parts" in data["published-print"]:
        year = str(data["published-print"]["date-parts"][0][0])
    elif "published-online" in data and "date-parts" in data["published-online"]:
        year = str(data["published-online"]["date-parts"][0][0])
    elif "issued" in data and "date-parts" in data["issued"]:
        year = str(data["issued"]["date-parts"][0][0])

    return {
        "DOI": doi,
        "Title": title,
        "Authors": authors_str,
        "Year": year
    }

# ------------------------
# Read DOIs from input CSV
# ------------------------
def read_dois_from_csv(input_file):
    dois = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:  # skip empty rows
                dois.append(row[0].strip())
    return dois

# ------------------------
# Save list of DOI info to CSV
# ------------------------
def save_dois_to_csv(dois, output_file="doi_metadata.csv"):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["DOI","Title","Authors","Year"])
        writer.writeheader()
        for doi in tqdm(dois, desc="Fetching DOI metadata"):
            info = parse_doi_data(doi)
            writer.writerow(info)
    print(f"Saved DOI metadata to {output_file}")

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    input_csv = "citation_scores_10.1038_nature14236.csv"    # replace with your input CSV file
    output_csv = "doiNames.csv" # replace with desired output CSV file
    doi_list = read_dois_from_csv(input_csv)
    save_dois_to_csv(doi_list, output_csv)
