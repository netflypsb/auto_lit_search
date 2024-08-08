import os
import json
import requests
from crewai_tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the SERPER_API_KEY from environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is not set in the environment variables.")

@tool("custom_google_scholar_search")
def custom_google_scholar_search(query: str) -> str:
    """Searches Google Scholar for scholarly articles. Provide it with the 
    relevant search terms in string format as the input and it will return the 
    search results in string format. Useful for searching for scholarly articles"""
    
    url = "https://google.serper.dev/scholar"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    
    return response.text
