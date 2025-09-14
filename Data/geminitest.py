import requests

API_KEY = "AIzaSyCFWagVzNXHMNISBjQfSa6t-4s2wUCi3q0"
API_SECRET = "your_api_secret"  # if needed
BASE_URL = "https://api.gemini.com/v1"  # Gemini REST API base

# Example: check account balances (requires authentication)
url = f"{BASE_URL}/balances"
headers = {
    "X-GEMINI-APIKEY": API_KEY,
    # You may need additional headers or signed payloads depending on Gemini API docs
}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("API key is valid!")
        print(response.json())  # account balances or returned data
    else:
        print(f"Error: {response.status_code}, {response.text}")
except Exception as e:
    print(f"Exception: {e}")
