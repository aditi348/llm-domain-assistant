import os
from dotenv import load_dotenv
import requests

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost",
    "X-Title": "LLM Domain Assistant"
}

def generate_answer(messages, temperature=0.5, model="mistralai/mistral-7b-instruct"):
    """
    messages: list of dicts with 'role' and 'content', e.g.
    [{"role": "system", "content": "You are ..."}, {"role": "user", "content": "Hello"}]
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.status_code} - {response.text}"







