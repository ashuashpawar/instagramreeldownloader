import pyaudio
import numpy as np
from openwakeword.model import Model
import subprocess
import speech_recognition as sr
import requests
import threading
import time
import os
import sys
sys.path.insert(0, os.path.expanduser("~/claude-agent"))

print("Loading wake word model...")
oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print("Hey Dev is ready! Say 'Hey Jarvis' to activate.")
subprocess.Popen(["say", "-v", "Samantha", "Hey Dev is ready!"])

is_responding = False
last_detected = 0

def execute_action(text):
    t = text.lower()
    if any(w in t for w in ["open chrome", "open browser", "open google"]):
        subprocess.Popen(["open", "-a", "Google Chrome"])
        return "Opening Chrome!"
    elif "open safari" in t:
        subprocess.Popen(["open", "-a", "Safari"])
        return "Opening Safari!"
    elif "open spotify" in t:
        subprocess.Popen(["open", "-a", "Spotify"])
        return "Opening Spotify!"
    elif "search for" in t and "instagram" in t:
        query = t.replace("search for","").replace("in instagram","").replace("on instagram","").replace("instagram","").strip()
        subprocess.Popen(["open", f"https://www.instagram.com/{query.replace(' ','').lower()}"])
        return f"Opening Instagram profile {query}"
    elif "search for" in t:
        query = t.replace("search for","").strip()
        subprocess.Popen(["open", f"https://www.google.com/search?q={query.replace(' ','+')}"])
        return f"Searching for {query}"
    elif "google" in t and "search" in t:
        query = t.replace("google","").replace("search","").strip()
        subprocess.Popen(["open", f"https://www.google.com/search?q={query.replace(' ','+')}"])
        return f"Searching Google for {query}"
    elif "open instagram" in t:
        subprocess.Popen(["open", "https://www.instagram.com"])
        return "Opening Instagram!"
    elif "take screenshot" in t:
        subprocess.Popen(["screencapture", "-i", "/tmp/screenshot.png"])
        return "Screenshot taken!"
    elif any(w in t for w in ["stop", "close", "quit", "goodbye", "bye"]):
        subprocess.Popen(["say", "-v", "Samantha", "Goodbye!"])
        time.sleep(2)
        os._exit(0)
    elif "close chrome" in t or "close browser" in t:
        subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to quit'])
        return "Chrome closed!"
    elif "close instagram" in t:
        subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to quit'])
        return "Closed!"
    elif any(w in t for w in ["what time", "current time"]):
        return f"It is {time.strftime('%I:%M %p')}"
    elif any(w in t for w in ["what date", "today's date"]):
        return f"Today is {time.strftime('%B %d %Y')}"
    return None

def get_ai_response(text):
    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llama3.2",
                  "prompt":f"You are Dev, a personal AI assistant. Your name is Dev, not Jarvis or Alexa. Be very concise, max 2 sentences.\nUser: {text}\nDev:",
                  "stream":False}, timeout=30)
        return r.json()["response"].strip()
    except Exception as e:
        return f"Sorry, I had an error: {str(e)}"

def listen_and_respond():
    global is_responding
    is_responding = True
    subprocess.Popen(["say", "-v", "Samantha", "Yes?"])
    time.sleep(0.8)
    print("Listening for command...")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")

        action_reply = execute_action(text)
        if action_reply:
            reply = action_reply
        else:
            reply = get_ai_response(text)

        print(f"Dev: {reply}")
        say = subprocess.Popen(["say", "-v", "Samantha", reply[:300]])
        say.wait()
    except sr.WaitTimeoutError:
        subprocess.Popen(["say", "-v", "Samantha", "I didn't catch that"])
    except sr.UnknownValueError:
        subprocess.Popen(["say", "-v", "Samantha", "Could not understand"])
    except Exception as e:
        print(f"Error: {e}")
    is_responding = False

audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

while True:
    data = stream.read(1280, exception_on_overflow=False)
    chunk = np.frombuffer(data, dtype=np.int16)
    oww.predict(chunk)
    score = oww.prediction_buffer["hey_jarvis"][-1]
    now = time.time()
    if score > 0.5 and not is_responding and (now - last_detected) > 3:
        last_detected = now
        print(f"\nWake word detected! Score: {score:.2f}")
        oww.prediction_buffer["hey_jarvis"].clear()
        threading.Thread(target=listen_and_respond, daemon=True).start()
