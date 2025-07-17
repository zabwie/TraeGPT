import requests
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "HammerAI/mistral-nemo-uncensored"
HISTORY_FILE = "chat_history.json"
MAX_TURNS = 25  # last 25 exchanges

SYSTEM_PROMPT = (
    "You are Trae, an empathetic, friendly, and helpful AI assistant. "
    "Keep your language natural, casual, and straightforward. "
    "Respond with short, clear sentences that feel like chatting with a close friend. "
    "Show empathy and understanding. "
    "Avoid giving medical, legal, or financial advice. "
    "If you don't know something, admit it honestly and simply. "
    "Match the user's tone—be chill if they're chill, serious if they're serious. "
    "Try to be helpful and answer every appropriate question. "
    "Use emojis sparingly, only when it feels natural. "
    "Above all, make the user feel heard and comfortable, like they're talking to a human, not a machine.\n\n"
)

API_KEY = "AIzaSyAodUqbh5-_2NxYCiq7LIN0UceHygIeUaw"
CSE_ID = "d384d627840d14bc2"

def google_search(query, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": API_KEY,
        "cx": CSE_ID,
        "num": num_results
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("items", []):
        results.append(f"{item['title']}\n{item['link']}\n{item['snippet']}\n")
    return "\n".join(results)

# Set up a requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Could not save history: {e}")

def load_history():
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Warning] Could not load history: {e}")
        return []

def chat():
    print("ChatGPT (HammerAI/mistral-nemo-uncensored) - Type 'exit' to quit.")
    history = load_history()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Handle web search command
        if user_input.startswith("search:"):
            query = user_input[len("search:"):].strip()
            try:
                search_result = google_search(query)
                print("\U0001F50E Web search results:\n")
                print(search_result)
            except Exception as e:
                print(f"[Search error] {e}")
            continue

        # Truncate history for prompt
        trimmed_history = history[-MAX_TURNS:]
        prompt = SYSTEM_PROMPT
        for turn in trimmed_history:
            prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
        prompt += f"User: {user_input}\nAI:"

        start_time = time.time()
        try:
            with session.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                print("AI: ", end="", flush=True)
                ai_output = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get("response", "")
                        print(chunk, end="", flush=True)
                        ai_output += chunk
                    except Exception as e:
                        print(f"\n[Stream error] {e}")
                print()  # Newline after streaming
        except Exception as e:
            print(f"[Error] {e}")
            continue
        end_time = time.time()
        print(f"[Response time: {end_time - start_time:.2f} seconds]")

        history.append({
            "user": user_input,
            "ai": ai_output.strip(),
            "timestamp": datetime.datetime.now().isoformat()
        })
        save_history(history)

if __name__ == "__main__":
    chat()
