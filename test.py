import os
import requests

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def openrouter_chat(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "moonshotai/kimi-k2:free",
        "messages": [
            {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# Example usage:
result = openrouter_chat("Hello!")
print(result)