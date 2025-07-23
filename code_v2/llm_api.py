# llm_api.py

import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: Set your API key in a .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = "http://localhost:3000"
YOUR_APP_NAME = "HPC-Analyzer"

def call_llm(prompt: str) -> str:
    """
    Calls the OpenRouter API with the given prompt.
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set. Please create a .env file."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/gpt-4o-mini", # Or any other model you prefer
        "messages": [{"role": "user", "content": prompt}],
        # 1. Set temperature to its lowest value to minimize randomness.
        "temperature": 0.0,
        # 2. Set top_p to its lowest value to disable nucleus sampling.
        "top_p": 0.0,
        
        # 3. Provide a fixed seed to guarantee identical output for the same prompt.
        "seed": 42
    }

    try:
        print("Calling LLM for analysis...")
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content

    except requests.exceptions.HTTPError as http_err:
        print(f"API request error: {http_err}")
        return "Error: Could not get a response from the API."
    except Exception as e:
        print(f"An unexpected error occurred in call_llm: {e}")
        return "Error: An unexpected error occurred."