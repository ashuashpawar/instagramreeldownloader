from flask import Flask, jsonify, request
import json, os, subprocess, threading, anthropic
from datetime import datetime

app = Flask(__name__)
API_KEY = open(os.path.expanduser("~/claude-agent/.apikey")).read().strip()
HISTORY_FILE = os.path.expanduser("~/claude-agent/history.json")
LEARNING_FILE = os.path.expanduser("~/claude-agent/learnings.json")
PROFILE_FILE = os.path.expanduser("~/claude-agent/profile.json")
HTML_FILE = "/tmp/chat_ui_template.html"
client = anthropic.Anthropic(api_key=API_KEY)

def load_json(path):
    try:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    except: pass
    return []

def save_json(path, data):
    try:
        with open(path,'w') as f: json.dump(data, f, indent=2)
    except: pass

def get_learning_context():
    try:
        profile = load_json(PROFILE_FILE)
        if not profile: return ""
        failures = profile.get("failure_patterns",[])[-5:]
        successes = profile.get("success_patterns",[])[-5:]
        common = profile.get("common_tasks",[])[-3:]
        if not any([failures,successes,common]): return ""
        ctx = "PAST LEARNINGS:\n"
        if common: ctx += "Common: "+"; ".join(common)+"\n"
        if successes: ctx += "Worked: "+"; ".join(successes)+"\n"
        if failures: ctx += "Failed: "+"; ".join(failures)+"\n"
        return ctx+"\n"
    except (json.JSONDecodeError, OSError, AttributeError):
        return ""

@app.route('/')
def index():
    try:
        return open(HTML_FILE).read()
    except OSError:
        return "<h1>Chat UI template not found</h1><p>Expected at: " + HTML_FILE + "</p>", 500

@app.route('/ask', methods=['POST'])
def ask():
    text = request.json.get('text','')
    try:
        history = load_json(HISTORY_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        history.append({"role":"user","content":text})
        import requests as req
        prompt = f"System: You are a helpful personal assistant. Today is {timestamp}. Be concise.\n\n"
        for m in history[-20:]:
            role = "User" if m["role"] == "user" else "Assistant"
            prompt += f"{role}: {m['content']}\n"
        prompt += "Assistant:"
        r = req.post("http://localhost:11434/api/generate",
            json={"model":"llama3.2","prompt":prompt,"stream":False}, timeout=120)
        reply = r.json()["response"].strip()
        history.append({"role":"assistant","content":reply})
        save_json(HISTORY_FILE, history[-50:])
        threading.Thread(target=lambda: subprocess.Popen(["say","-v","Samantha",reply[:200]]), daemon=True).start()
        return jsonify({"reply":reply})
    except Exception as e:
        return jsonify({"reply":f"Error: {str(e)}"})

@app.route('/screen')
def screen():
    try:
        import pyautogui, pytesseract
        from PIL import Image
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((screenshot.width//2, screenshot.height//2))
        screenshot.save("/tmp/claude_screen.png", optimize=True, quality=60)
        text = pytesseract.image_to_string(Image.open("/tmp/claude_screen.png"))
        if text.strip():
            history = load_json(HISTORY_FILE)
            history.append({"role":"user","content":f"Screen shows:\n{text[:2000]}\nSummarize briefly."})
            prompt2 = "You are a helpful assistant.\n\n" + "\n".join([("User: " if m["role"]=="user" else "Assistant: ") + m["content"] for m in history[-10:]]) + "\nAssistant:"
            reply = req.post("http://localhost:11434/api/generate",
                json={"model":"llama3.2","prompt":prompt2,"stream":False}, timeout=120).json()["response"].strip()
            history.append({"role":"assistant","content":reply})
            save_json(HISTORY_FILE, history[-50:])
            return jsonify({"reply":reply})
        return jsonify({"reply":"Could not read screen text."})
    except Exception as e:
        return jsonify({"reply":f"Error: {str(e)}"})

@app.route('/task', methods=['POST'])
def task():
    import pyautogui, base64, time
    task_desc = request.json.get('text','')
    log = []
    try:
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((screenshot.width//2, screenshot.height//2))
        screenshot.save("/tmp/claude_screen.png", optimize=True, quality=60)
        with open("/tmp/claude_screen.png","rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")
        screen_w, screen_h = pyautogui.size()
        plan_prompt = f"{get_learning_context()}You are controlling a Mac. Screen: {screen_w}x{screen_h}.\nTask: {task_desc}\nReply ONLY with JSON array using actions: open_url, wait, click, type, press, screenshot. JSON only."
        plan_reply = req.post("http://localhost:11434/api/generate",
            json={"model":"llama3.2","prompt":plan_prompt,"stream":False}, timeout=120).json()["response"].strip()
        try:
            steps = json.loads(plan_reply.replace("```json","").replace("```","").strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ollama returned invalid task plan (not JSON): {e}\n\nRaw: {plan_reply[:500]}")
        if not isinstance(steps, list):
            raise RuntimeError(f"Unexpected plan format (expected list, got {type(steps).__name__})")
        log.append(f"Plan: {len(steps)} steps")
        for step in steps:
            action = step.get("action")
            value = step.get("value","")
            if action == "open_url":
                subprocess.Popen(["open", value])
                log.append(f"Opened {value}")
                time.sleep(4)
                subprocess.run(["osascript","-e",'tell application "Google Chrome" to activate'],capture_output=True)
                time.sleep(1)
            elif action == "wait":
                try:
                    time.sleep(float(value))
                    log.append(f"Waited {value}s")
                except ValueError:
                    log.append(f"Skipped invalid wait value: '{value}'")
            elif action == "click":
                screenshot2 = pyautogui.screenshot()
                screenshot2 = screenshot2.resize((screenshot2.width//2, screenshot2.height//2))
                screenshot2.save("/tmp/claude_screen.png", optimize=True, quality=60)
                with open("/tmp/claude_screen.png","rb") as f:
                    img2 = base64.standard_b64encode(f.read()).decode("utf-8")
                click_prompt = f"Find '{value}' on screen. Reply ONLY CLICK:x,y or NOTFOUND."
                rt = req.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2","prompt":click_prompt,"stream":False}, timeout=60).json()["response"].strip()
                if rt.startswith("CLICK:"):
                    try:
                        x,y = map(int, rt.replace("CLICK:","").split(","))
                        pyautogui.moveTo(x*2,y*2,duration=0.4)
                        pyautogui.click(x*2,y*2)
                        log.append(f"Clicked '{value}'")
                    except (ValueError, IndexError) as e:
                        log.append(f"Could not parse click coords for '{value}' - skipped")
                    time.sleep(0.8)
                else:
                    log.append(f"Could not find '{value}' - skipped")
            elif action == "type":
                subprocess.run(["osascript","-e",f'set the clipboard to "{value}"'],capture_output=True)
                time.sleep(1)
                subprocess.run(["osascript","-e",'tell application "Google Chrome" to activate\ndelay 0.5\ntell application "System Events" to keystroke "v" using command down'],capture_output=True)
                log.append(f"Typed: {value}")
                time.sleep(0.3)
            elif action == "press":
                pyautogui.press(value)
                log.append(f"Pressed: {value}")
                time.sleep(0.8)
            elif action == "screenshot":
                pyautogui.screenshot().save("/tmp/claude_screen.png")
                subprocess.Popen(["open","/tmp/claude_screen.png"])
                log.append("Screenshot taken")
                time.sleep(1)
        return jsonify({"reply":"Task done!\n\n"+"\n".join(f"- {l}" for l in log)})
    except Exception as e:
        return jsonify({"reply":f"Task error: {str(e)}\n\n"+"\n".join(f"- {l}" for l in log)})

@app.route('/stop')
def stop():
    subprocess.run(["killall","say"],capture_output=True)
    return jsonify({"ok":True})

@app.route('/history')
def history():
    return jsonify(load_json(HISTORY_FILE))

@app.route('/learnings')
def learnings():
    return jsonify(load_json(LEARNING_FILE))

@app.route('/profile')
def profile():
    return jsonify(load_json(PROFILE_FILE))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    save_json(HISTORY_FILE,[]); return jsonify({"ok":True})

@app.route('/clear_learnings', methods=['POST'])
def clear_learnings():
    save_json(LEARNING_FILE,[]); return jsonify({"ok":True})

@app.route('/delete_history', methods=['POST'])
def delete_history():
    idx = request.json.get('index',-1)
    h = load_json(HISTORY_FILE)
    hr = list(reversed(h))
    if 0 <= idx < len(hr): hr.pop(idx)
    save_json(HISTORY_FILE, list(reversed(hr)))
    return jsonify({"ok":True})

if __name__ == '__main__':
    print("Chat UI running at http://localhost:5005")
    subprocess.Popen(["open","http://localhost:5005"])
    app.run(port=5005, debug=False)
