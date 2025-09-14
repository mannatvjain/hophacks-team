import pandas as pd
import requests
import time

def get_abstract_from_crossref(doi):
    """
    Fetch abstract from CrossRef API for a given DOI.
    Returns abstract string or empty string if not found.
    """
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # The abstract is often in data['message']['abstract']
        abstract = data['message'].get('abstract', '')
        return abstract
    except Exception as e:
        print(f"Could not fetch abstract for DOI {doi}: {e}")
        return ''

def add_abstracts_to_csv(input_file, output_file, delay=1):
    """
    Open CSV, fetch abstracts for each DOI, and save new CSV with 'Abstract' column.
    delay: seconds to wait between API requests to avoid rate limits
    """
    df = pd.read_csv(input_file)

    abstracts = []
    for i, doi in enumerate(df['DOI']):
        print(f"Fetching abstract {i+1}/{len(df)}: {doi}")
        abstract = get_abstract_from_crossref(doi)
        abstracts.append(abstract)
        time.sleep(delay)  # polite delay to avoid hammering API

    df['Abstract'] = abstracts
    df.to_csv(output_file, index=False)
    print(f"Saved CSV with abstracts to {output_file}")

if __name__ == "__main__":
    input_file = "final_merged.csv"
    output_file = "final_merged_with_abstracts.csv"

    add_abstracts_to_csv(input_file, output_file)
