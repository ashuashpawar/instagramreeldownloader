import requests
import json

def chat(messages, system="", model="llama3.2"):
    prompt = ""
    if system:
        prompt += f"System: {system}\n\n"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"
    prompt += "Assistant:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response format: {list(data.keys())}")
        return data["response"].strip()
    except requests.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running? (ollama serve)")
    except requests.Timeout:
        raise RuntimeError("Ollama request timed out after 120s")
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama returned HTTP error: {e}")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Ollama returned invalid response: {e}")
