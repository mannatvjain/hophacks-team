import requests
from bs4 import BeautifulSoup

def get_doi_from_nature(url):
    """Scrape a Nature article page to find its DOI."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    
    meta_tag = soup.find("meta", {"name": "citation_doi"})
    if meta_tag and "content" in meta_tag.attrs:
        return meta_tag["content"]
    return None

def get_citations_opencitations(doi):
    """Fetch all citing papers from OpenCitations and return their DOIs as URLs."""
    api_url = f"https://opencitations.net/index/coci/api/v1/citations/{doi}"
    r = requests.get(api_url).json()
    
    citing_dois = [entry["citing"] for entry in r if "citing" in entry]
    citing_links = [f"https://doi.org/{d}" for d in citing_dois]
    return citing_links

# Example Nature article
url = "https://www.nature.com/articles/547156a"

doi = get_doi_from_nature(url)
if doi:
    print(f"DOI found: {doi}\n")
    
    citations = get_citations_opencitations(doi)
    print(f"Found {len(citations)} citations:\n")
    
    for link in citations:
        print(link)
else:
    print("DOI not found on page.")
