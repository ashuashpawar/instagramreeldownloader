from deepface import DeepFace
import cv2
import time
import threading
import subprocess
import speech_recognition as sr
import requests

current_mood = "neutral"
is_listening = False

def get_ai_response(text, mood):
    prompt = f"""You are an empathetic AI assistant. 
The user's current detected mood is: {mood}
Respond appropriately to their mood - if happy be upbeat, if sad be supportive, if angry be calm, if stressed be reassuring.
Keep response under 3 sentences.

User says: {text}"""
    
    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llama3.2","prompt":prompt,"stream":False}, timeout=30)
        return r.json()["response"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def speak(text):
    subprocess.Popen(["say", "-v", "Samantha", text])

def listen_and_respond():
    global is_listening, current_mood
    is_listening = True
    print("\n🎙️ Listening...")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        print(f"Your mood: {current_mood}")
        print("AI thinking...")
        reply = get_ai_response(text, current_mood)
        print(f"AI: {reply}")
        speak(reply)
    except Exception as e:
        print(f"Error: {e}")
    is_listening = False

print("Mood Chat started!")
print("Press SPACE to talk, Q to quit\n")

cap = cv2.VideoCapture(0)
last_check = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_check > 3:
        last_check = now
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                enforce_detection=False, silent=True)
            current_mood = result[0]['dominant_emotion']
        except:
            pass

    color = {
        'happy': (0,255,0), 'sad': (255,0,0),
        'angry': (0,0,255), 'fear': (255,165,0),
        'surprise': (255,255,0), 'neutral': (200,200,200),
        'disgust': (128,0,128)
    }.get(current_mood, (200,200,200))

    cv2.putText(frame, f"Mood: {current_mood.upper()}", (20,40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    status = "Listening..." if is_listening else "SPACE=Talk  Q=Quit"
    cv2.putText(frame, status, (20,80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow('Mood Chat - Talk to AI', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not is_listening:
        threading.Thread(target=listen_and_respond, daemon=True).start()

cap.release()
cv2.destroyAllWindows()
print("Goodbye!")
