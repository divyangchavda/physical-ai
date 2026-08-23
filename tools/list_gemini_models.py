"""List available Gemini models."""
import os
import sys

api_key = sys.argv[1] if len(sys.argv) > 1 else input("API key: ").strip()

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("Available Gemini models:\n")
    models = client.models.list()
    for model in models:
        print(f"  - {model.name}")
            
except Exception as e:
    print(f"Error: {e}")
