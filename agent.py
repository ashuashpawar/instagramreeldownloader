import os
import sys
sys.path.insert(0, os.path.expanduser("~/claude-agent"))
import rumps
import anthropic
import ollama_client
import video_editor
sys.path.insert(0, os.path.expanduser("~/claude-agent"))
import importlib
import instagram_tools
importlib.reload(instagram_tools)
import threading
import subprocess
import speech_recognition as sr
import pyautogui
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup
import json
import os
import base64
import time
import re as _re
from collections import deque
from datetime import datetime

USE_OLLAMA = True  # True = use Ollama for text, Claude only for vision

def needs_claude(text):
    """Returns True only if task needs vision/complex reasoning"""
    vision_keywords = ["task", "click", "screen", "find", "open", "type", "press", "crawl", "website"]
    return any(k in text.lower() for k in vision_keywords)

def smart_chat(messages, system="", user_input=""):
    """Auto picks Ollama for simple chat, Claude for vision/complex tasks"""
    if USE_OLLAMA and not needs_claude(user_input):
        return ollama_client.chat(messages=messages, system=system)
    else:
        import anthropic as _anth
        _client = _anth.Anthropic(api_key=API_KEY)
        for attempt in range(3):
            try:
                result = _client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=1024,
                    system=system,
                    messages=messages
                )
                return result.content[0].text
            except _anth.APIStatusError as e:
                if e.status_code == 529 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
API_KEY = "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"
HISTORY_FILE    = os.path.expanduser("~/claude-agent/history.json")
LEARNING_FILE   = os.path.expanduser("~/claude-agent/learnings.json")
PROFILE_FILE    = os.path.expanduser("~/claude-agent/profile.json")
WORKFLOWS_FILE  = os.path.expanduser("~/claude-agent/learned_workflows.json")

_OPTION_NAMES = {
    1:"Add Captions", 2:"Auto Captions (AI)", 3:"Trim Video", 4:"Merge Videos",
    5:"Add Background Music", 6:"Extract Audio", 7:"Change Speed", 8:"Fade In/Out",
    9:"Add Intro/Outro", 10:"Resize for Platform", 11:"Remove Background",
    12:"Change Background", 13:"Chroma Key", 14:"Portrait Blur",
    15:"Face Blur", 16:"Face Pixelate", 17:"Face Swap", 18:"Install Face Swap GAN",
    19:"Enhance Faces", 20:"Setup Face Enhancer", 21:"Install GFPGAN",
    22:"Color Grade", 23:"Add Watermark", 24:"Lower Third", 25:"Picture-in-Picture",
    26:"SRT Subtitles", 27:"Add Transition", 28:"Zoom Effect",
    29:"Ken Burns", 30:"AI Voiceover", 31:"Script to Video", 32:"AI Avatar",
    33:"Slideshow", 34:"Extract Frames", 35:"Stabilize Video",
}

class ClaudeAgent(rumps.App):
    def __init__(self):
        super(ClaudeAgent, self).__init__("🤖")
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.history = self.load_history()
        self.voice_output = True
        self.say_process = None
        # ── Workflow recorder ──────────────────────────────────────────
        self._recording      = False   # True while we're recording a workflow
        self._session_steps  = []      # steps captured so far
        self._step_inputs    = []      # inputs collected for the current step
        self._step_num       = None    # option number of current step
        # ── Workflow replayer ──────────────────────────────────────────
        self._replaying      = False
        self._replay_queue   = deque() # pre-loaded inputs for the current step
        self.menu = [
            "Ask Claude",
            "Ask by Voice",
            "Talk to Me",
            "Mood Chat",
            None,
            "Do Task",
            "Read My Screen",
            "Smart Click",
            "Type Anywhere",
            "Open Website",
            "Crawl Website",
            "Open Chat UI",
            "Search Profile",
            "Download Reel",
            "Post Reel",
            "Download User Reels",
            "Video Editor",
            None,
            "Stop Speaking",
            "Voice Output: ON",
            "Clear History",
            None
        ]

    def load_history(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r") as f:
                    data = json.load(f)
                    return data[-20:]
        except json.JSONDecodeError as e:
            print(f"Warning: history file corrupted, starting fresh ({e})")
        except OSError as e:
            print(f"Warning: could not read history ({e})")
        return []

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.history[-50:], f, indent=2)
        except OSError as e:
            print(f"Warning: could not save history ({e})")


    def save_learning(self, task, log):
        try:
            learnings = []
            if os.path.exists(LEARNING_FILE):
                with open(LEARNING_FILE, "r") as f:
                    learnings = json.load(f)
            entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "task": task,
                "successes": [s for s in log if "Clicked" in s or "Typed" in s or "Opened" in s],
                "failures": [s for s in log if "skipped" in s or "error" in s.lower()],
                "total_steps": len(log)
            }
            learnings.append(entry)
            with open(LEARNING_FILE, "w") as f:
                json.dump(learnings[-200:], f, indent=2)
            self.update_profile(task, entry)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not save learning ({e})")

    def update_profile(self, task, entry):
        try:
            profile = {"style": {}, "common_tasks": [], "failure_patterns": [], "success_patterns": []}
            if os.path.exists(PROFILE_FILE):
                with open(PROFILE_FILE, "r") as f:
                    profile = json.load(f)
            if task not in profile["common_tasks"]:
                profile["common_tasks"].append(task)
            profile["common_tasks"] = profile["common_tasks"][-50:]
            for s in entry["successes"]:
                if s not in profile["success_patterns"]:
                    profile["success_patterns"].append(s)
            profile["success_patterns"] = profile["success_patterns"][-100:]
            for s in entry["failures"]:
                if s not in profile["failure_patterns"]:
                    profile["failure_patterns"].append(s)
            profile["failure_patterns"] = profile["failure_patterns"][-100:]
            with open(PROFILE_FILE, "w") as f:
                json.dump(profile, f, indent=2)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not update profile ({e})")

    def get_learning_context(self):
        try:
            context = ""
            if os.path.exists(PROFILE_FILE):
                with open(PROFILE_FILE, "r") as f:
                    profile = json.load(f)
                failures = profile.get("failure_patterns", [])[-10:]
                successes = profile.get("success_patterns", [])[-10:]
                common = profile.get("common_tasks", [])[-5:]
                if failures or successes or common:
                    context += "PAST LEARNINGS:\n"
                    if common:
                        context += "Common tasks: " + "; ".join(common[-3:]) + "\n"
                    if successes:
                        context += "What worked: " + "; ".join(successes[-5:]) + "\n"
                    if failures:
                        context += "What failed (avoid these): " + "; ".join(failures[-5:]) + "\n"
                    context += "\n"
            return context
        except (json.JSONDecodeError, OSError):
            return ""

    def show_response(self, reply):
        # ── Capture output path for recorder & replay chaining ────────
        m = _re.search(r'Saved:\s*(.+\.(?:mp4|mov|jpg|jpeg|png|mp3|wav|m4a))',
                       reply, _re.IGNORECASE)
        out_file = m.group(1).strip() if m else None
        self._last_output_path = out_file   # used by replay chainer

        if self._recording and self._step_num is not None:
            self._end_step(out_file)

        with open("/tmp/claude_reply.txt", "w") as f:
            f.write(reply)
        script = '''
set filePath to "/tmp/claude_reply.txt"
set fileRef to open for access POSIX file filePath
set msg to read fileRef
close access fileRef
display dialog msg with title "Claude Agent" buttons {"OK", "Copy", "Stop"} default button "OK"
set btn to button returned of result
if btn is "Copy" then
    set the clipboard to msg
end if
if btn is "Stop" then
    do shell script "killall say 2>/dev/null; true"
end if
'''
        subprocess.run(["osascript", "-e", script])

    # ── Workflow recording helpers ─────────────────────────────────────

    def _begin_step(self, option_num):
        """Called at the start of each video editor option."""
        self._step_num    = option_num
        self._step_inputs = []

    def _end_step(self, output_path):
        """Called when show_response fires — finalise and save the step."""
        if self._step_num is None:
            return
        self._session_steps.append({
            'option' : self._step_num,
            'name'   : _OPTION_NAMES.get(self._step_num, f'Option {self._step_num}'),
            'inputs' : list(self._step_inputs),
            'output' : output_path,
        })
        self._step_num    = None
        self._step_inputs = []

    def _load_workflows(self):
        try:
            if os.path.exists(WORKFLOWS_FILE):
                with open(WORKFLOWS_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_workflows(self, wf):
        with open(WORKFLOWS_FILE, 'w') as f:
            json.dump(wf, f, indent=2)

    def _run_saved_workflow(self, name):
        """Replay a saved workflow, asking for a fresh video for the first step."""
        wf = self._load_workflows()
        if name not in wf:
            self.show_response(f"Workflow '{name}' not found.")
            return

        steps = wf[name]['steps']
        prev_output = None   # output of the previous step

        for step in steps:
            option = step['option']
            inputs = step['inputs']

            # Build a queue of inputs for this step.
            # If an input is a video file AND we have a previous step output,
            # substitute it automatically (pipeline chaining).
            queue = deque()
            first_video_substituted = False
            for inp in inputs:
                item = dict(inp)  # copy
                if (inp.get('is_video') and prev_output and
                        not first_video_substituted):
                    # Chain: use previous step's output as this step's video input
                    item['replay_value'] = prev_output
                    first_video_substituted = True
                elif inp.get('is_video') and not first_video_substituted:
                    # First step — ask user to pick fresh video
                    fresh = self.mac_pick_file(
                        f"Select video for: {step['name']}",
                        file_types=["public.movie", "public.mpeg-4",
                                    "com.apple.quicktime-movie"],
                        _is_video=False)   # don't re-log during replay
                    item['replay_value'] = fresh
                    first_video_substituted = True
                queue.append(item)

            # Inject queue and run the option through the normal dispatch path
            self._replaying    = True
            self._replay_queue = queue
            self._step_num     = None  # don't re-record while replaying
            try:
                self._dispatch_video_option(option)
            finally:
                self._replaying    = False
                self._replay_queue = deque()

            # Grab the output of this step (last step added to session log won't
            # exist since we disabled recording; capture from the result text instead)
            # We track prev_output via a small intercept in show_response above —
            # but we're not recording, so we need another way.
            # Simplest: parse the result from the saved step's known output name.
            # For now rely on the fact that video_editor uses out_path which saves
            # to OUTPUT folder — we note output from the last result shown.
            # (prev_output updated in _replay_capture below)
            prev_output = getattr(self, '_last_output_path', None)

    def speak(self, text):
        if self.voice_output:
            with open("/tmp/claude_speak.txt", "w") as f:
                f.write(text[:800])
            self.say_process = subprocess.Popen(["say", "-f", "/tmp/claude_speak.txt", "-v", "Samantha"])

    def stop_speaking(self):
        try:
            if self.say_process:
                self.say_process.terminate()
                self.say_process = None
            subprocess.run(["killall", "say"], capture_output=True)
        except:
            pass

    def mac_input(self, prompt):
        # Replay mode: return saved value instead of showing dialog
        if self._replaying and self._replay_queue:
            item = self._replay_queue[0]
            if item.get('kind') == 'text':
                self._replay_queue.popleft()
                return item['value']

        with open("/tmp/claude_prompt.txt", "w") as f:
            f.write(prompt)
        script = '''
set filePath to "/tmp/claude_prompt.txt"
set fileRef to open for access POSIX file filePath
set msg to read fileRef
close access fileRef
set dlg to display dialog msg default answer "" with title "Claude Agent" buttons {"Cancel", "OK"} default button "OK"
return text returned of dlg
'''
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True)
        if result.returncode != 0:
            return None
        text = result.stdout.strip()
        value = text if text else None

        # Record mode: log this input
        if self._recording and self._step_num is not None:
            self._step_inputs.append({'kind': 'text', 'prompt': prompt, 'value': value})

        return value

    def mac_pick_file(self, prompt="Select a file:", file_types=None, _is_video=False):
        """Open a native macOS file-chooser dialog. Returns POSIX path or None."""
        # Replay mode: return saved value
        if self._replaying and self._replay_queue:
            item = self._replay_queue[0]
            if item.get('kind') == 'file':
                self._replay_queue.popleft()
                # If this is a video that was chained from previous step output, use substituted value
                return item.get('replay_value', item['value'])

        with open("/tmp/claude_prompt.txt", "w") as f:
            f.write(prompt)
        if file_types:
            type_list = '{' + ', '.join(f'"{t}"' for t in file_types) + '}'
            type_clause = f'of type {type_list}'
        else:
            type_clause = ''
        script = f'''
set msg to (read POSIX file "/tmp/claude_prompt.txt")
try
    set f to choose file with prompt msg {type_clause}
    return POSIX path of f
on error
    return ""
end try
'''
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True)
        path = result.stdout.strip()
        value = path if path else None

        # Record mode: log this input
        if self._recording and self._step_num is not None:
            self._step_inputs.append({'kind': 'file', 'prompt': prompt,
                                      'value': value, 'is_video': _is_video})
        return value

    def mac_pick_folder(self, prompt="Select a folder:"):
        """Open a native macOS folder-chooser dialog. Returns POSIX path or None."""
        # Replay mode
        if self._replaying and self._replay_queue:
            item = self._replay_queue[0]
            if item.get('kind') == 'folder':
                self._replay_queue.popleft()
                return item['value']

        with open("/tmp/claude_prompt.txt", "w") as f:
            f.write(prompt)
        script = '''
set msg to (read POSIX file "/tmp/claude_prompt.txt")
try
    set f to choose folder with prompt msg
    return POSIX path of f
on error
    return ""
end try
'''
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True)
        path = result.stdout.strip()
        value = path.rstrip('/') if path else None

        if self._recording and self._step_num is not None:
            self._step_inputs.append({'kind': 'folder', 'prompt': prompt, 'value': value})
        return value

    def take_screenshot_b64(self):
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((screenshot.width // 2, screenshot.height // 2))
        screenshot.save("/tmp/claude_screen.png", optimize=True, quality=60)
        with open("/tmp/claude_screen.png", "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def find_and_click(self, what, retries=3):
        screen_w, screen_h = pyautogui.size()
        for attempt in range(retries):
            img_data = self.take_screenshot_b64()
            try:
                result = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=64,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
                            {"type": "text", "text": f"Screenshot of Mac screen ({screen_w}x{screen_h}). Find: '{what}'. Reply ONLY with CLICK:x,y or NOTFOUND."}
                        ]
                    }]
                )
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < retries - 1:
                    time.sleep(10)
                    continue
                raise
            response_text = result.content[0].text.strip()
            if response_text.startswith("CLICK:"):
                try:
                    coords = response_text.replace("CLICK:", "").strip()
                    x, y = map(int, coords.split(","))
                    x = max(0, min(x, screen_w))
                    y = max(0, min(y, screen_h))
                    pyautogui.moveTo(x, y, duration=0.4)
                    pyautogui.click(x, y)
                    return True, x, y
                except (ValueError, IndexError) as e:
                    print(f"Warning: could not parse click coordinates '{response_text}' ({e})")
            if attempt < retries - 1:
                time.sleep(2)
        return False, 0, 0

    def do_task(self, task_description):
        log = []
        try:
            self.title = "🧠"
            img_data = self.take_screenshot_b64()
            screen_w, screen_h = pyautogui.size()

            plan_result = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
                        {"type": "text", "text": f"""{self.get_learning_context()}You are controlling a Mac. Screen: {screen_w}x{screen_h}.
Task: {task_description}
Reply ONLY with JSON array:
[
  {{"action": "open_url", "value": "https://www.google.com"}},
  {{"action": "wait", "value": "4"}},
  {{"action": "click", "value": "Google search input box"}},
  {{"action": "wait", "value": "1"}},
  {{"action": "type", "value": "weather in Pune"}},
  {{"action": "wait", "value": "0.5"}},
  {{"action": "press", "value": "enter"}},
]
Actions: open_url, wait, click, type, press, screenshot, read_clipboard. Use read_clipboard after any copy action to capture the result. JSON only."""}
                    ]
                }]
            )

            plan_text = plan_result.content[0].text.strip()
            plan_text = plan_text.replace("```json", "").replace("```", "").strip()
            try:
                steps = json.loads(plan_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Claude returned invalid task plan (not JSON): {e}\n\nRaw response:\n{plan_text[:500]}")
            if not isinstance(steps, list):
                raise RuntimeError(f"Claude returned unexpected plan format (expected list, got {type(steps).__name__})")
            log.append(f"Plan: {len(steps)} steps")

            for step in steps:
                action = step.get("action")
                value = step.get("value", "")
                self.title = "⚙️"

                if action == "open_url":
                    subprocess.Popen(["open", value])
                    log.append(f"Opened {value}")
                    time.sleep(4)
                    subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to activate'], capture_output=True)
                    time.sleep(1)

                elif action == "wait":
                    try:
                        time.sleep(float(value))
                        log.append(f"Waited {value}s")
                    except ValueError:
                        log.append(f"Skipped invalid wait value: '{value}'")

                elif action == "click":
                    found, x, y = self.find_and_click(value)
                    if found:
                        log.append(f"Clicked '{value}' at {x},{y}")
                        time.sleep(0.8)
                    else:
                        log.append(f"Could not find '{value}' - skipped")

                elif action == "type":
                    subprocess.run(["osascript", "-e", f'set the clipboard to "{value}"'], capture_output=True)
                    time.sleep(1)
                    subprocess.run(["osascript", "-e", '''
tell application "Google Chrome" to activate
delay 0.5
tell application "System Events" to keystroke "v" using command down
'''], capture_output=True)
                    log.append(f"Typed: {value}")
                    time.sleep(0.3)

                elif action == "press":
                    pyautogui.press(value)
                    log.append(f"Pressed: {value}")
                    time.sleep(0.8)

                elif action == "screenshot":
                    screenshot = pyautogui.screenshot()
                    screenshot.save("/tmp/claude_screen.png")
                    subprocess.Popen(["open", "/tmp/claude_screen.png"])
                    log.append("Screenshot taken and opened")
                    time.sleep(1)

                elif action == "read_clipboard":
                    result = subprocess.run(["pbpaste"], capture_output=True, text=True)
                    clipboard_text = result.stdout.strip()
                    log.append(f"Clipboard: {clipboard_text}")

            summary = "Task completed!\n\n" + "\n".join(f"- {l}" for l in log)
            self.title = "🤖"
            self.save_learning(task_description, log)
            threading.Thread(target=self.speak, args=("Task completed!",), daemon=True).start()
            threading.Thread(target=self.show_response, args=(summary,), daemon=True).start()

        except Exception as e:
            self.title = "🤖"
            self.show_response(f"Task error: {str(e)}\n\nSteps so far:\n" + "\n".join(f"- {l}" for l in log))

    def record_and_transcribe(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
        return recognizer.recognize_google(audio)

    def handle_voice(self):
        try:
            self.title = "🎙️"
            self.show_response("Listening... speak after closing this!")
            text = self.record_and_transcribe()
            if text:
                self.title = "⏳"
                self.get_response(text)
        except sr.WaitTimeoutError:
            self.title = "🤖"
            self.show_response("No speech detected. Try again.")
        except sr.UnknownValueError:
            self.title = "🤖"
            self.show_response("Could not understand. Try again.")
        except Exception as e:
            err = str(e)
            if "overloaded" in err.lower() or "529" in err:
                self.show_response("Anthropic is busy, retrying in 10 seconds...")
                time.sleep(10)
                self.get_response(user_input)
            else:
                self.title = "🤖"
                self.show_response(f"Error: {err}")

    def do_screen_read(self):
        try:
            self.title = "📸"
            screenshot = pyautogui.screenshot()
            screenshot.save("/tmp/claude_screen.png")
            img = Image.open("/tmp/claude_screen.png")
            text = pytesseract.image_to_string(img)
            if text.strip():
                prompt = f"I took a screenshot of my Mac screen. Here is the text:\n\n{text[:3000]}\n\nSummarize what is on screen and suggest what I can do."
            else:
                prompt = "I took a screenshot but could not extract text."
            self.get_response(prompt)
        except Exception as e:
            self.title = "🤖"
            self.show_response(f"Screen read error: {str(e)}")

    def do_smart_click(self):
        try:
            user_input = self.mac_input("What do you want to click?")
            if not user_input:
                return
            self.title = "🔍"
            found, x, y = self.find_and_click(user_input)
            if found:
                self.title = "🤖"
                self.show_response(f"Clicked '{user_input}' at {x},{y}")
            else:
                self.title = "🤖"
                self.show_response(f"Could not find '{user_input}' on screen.")
        except Exception as e:
            self.title = "🤖"
            self.show_response(f"Smart click error: {str(e)}")

    def do_type_anywhere(self):
        try:
            user_input = self.mac_input("What do you want to type?")
            if user_input:
                time.sleep(1)
                subprocess.run(["osascript", "-e", f'set the clipboard to "{user_input}"'], capture_output=True)
                pyautogui.hotkey("command", "v")
                self.show_response(f"Typed: {user_input}")
        except Exception as e:
            self.show_response(f"Type error: {str(e)}")

    def do_open_website(self):
        try:
            result = subprocess.run(["osascript", "-e",
                'display dialog "Enter website URL:" default answer "https://" with title "Open Website" buttons {"Cancel","Open & Read"} default button "Open & Read"'],
                capture_output=True, text=True)
            if result.returncode == 0:
                url = result.stdout.split("text returned:")[1].strip()
                subprocess.Popen(["open", url])
                self.title = "🌐"
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                page_text = soup.get_text(separator="\n", strip=True)[:3000]
                prompt = f"I opened {url}. Content:\n\n{page_text}\n\nSummarize and highlight key info."
                self.get_response(prompt)
        except Exception as e:
            self.title = "🤖"
            self.show_response(f"Website error: {str(e)}")

    def get_response(self, user_input):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.history.append({"role": "user", "content": user_input})
            reply = smart_chat(
                messages=self.history[-20:],
                system=f"You are a personal AI agent on a Mac. Today is {timestamp}. When user asks you to DO something, always DO it using available tools - never just say yes or describe what you would do. Be concise. {self.get_learning_context()}",
                user_input=user_input
            )
            self.history.append({"role": "assistant", "content": reply})
            self.save_history()
            self.title = "🤖"
            threading.Thread(target=self.speak, args=(reply,), daemon=True).start()
            threading.Thread(target=self.show_response, args=(reply,), daemon=True).start()
        except Exception as e:
            err = str(e)
            if "overloaded" in err.lower() or "529" in err:
                self.show_response("Anthropic is busy, retrying in 10 seconds...")
                time.sleep(10)
                self.get_response(user_input)
            else:
                self.title = "🤖"
                self.show_response(f"Error: {err}")

    @rumps.clicked("Ask Claude")
    def ask_claude(self, _):
        user_input = self.mac_input("What do you want Claude to do?")
        if user_input and user_input.strip():
            self.title = "⏳"
            threading.Thread(target=self.get_response, args=(user_input.strip(),), daemon=True).start()

    @rumps.clicked("Ask by Voice")
    def ask_by_voice(self, _):
        threading.Thread(target=self.handle_voice, daemon=True).start()

    @rumps.clicked("Do Task")
    def do_task_click(self, _):
        user_input = self.mac_input("Describe the task (e.g. open google and search for weather in Pune)")
        if user_input and user_input.strip():
            threading.Thread(target=self.do_task, args=(user_input.strip(),), daemon=True).start()

    @rumps.clicked("Read My Screen")
    def read_screen(self, _):
        threading.Thread(target=self.do_screen_read, daemon=True).start()

    @rumps.clicked("Smart Click")
    def smart_click(self, _):
        threading.Thread(target=self.do_smart_click, daemon=True).start()

    @rumps.clicked("Type Anywhere")
    def type_anywhere(self, _):
        threading.Thread(target=self.do_type_anywhere, daemon=True).start()

    @rumps.clicked("Open Website")
    def open_website(self, _):
        threading.Thread(target=self.do_open_website, daemon=True).start()

    @rumps.clicked("Open Chat UI")
    def open_chat_ui(self, _):
        threading.Thread(target=self._start_chat_ui, daemon=True).start()

    def _start_chat_ui(self):
        import socket
        s = socket.socket()
        in_use = s.connect_ex(("localhost", 5005)) == 0
        s.close()
        if not in_use:
            subprocess.Popen(["python3", os.path.expanduser("~/claude-agent/chat_ui.py")])
            import time; time.sleep(1)
        subprocess.Popen(["open", "http://localhost:5005"])


    def do_crawl_website(self):
        try:
            url = self.mac_input("Enter website URL to crawl")
            if not url:
                return
            self.title = "🕷️"
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from urllib.parse import urljoin, urlparse
            from bs4 import BeautifulSoup
            import time
            domain = urlparse(url).netloc
            visited = set()
            all_content = []
            to_visit = [url]
            max_pages = 15
            opts = Options()
            opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
            try:
                while to_visit and len(visited) < max_pages:
                    current = to_visit.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    try:
                        driver.get(current)
                        time.sleep(3)
                        soup = BeautifulSoup(driver.page_source, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        title = soup.title.string if soup.title else "No title"
                        text = soup.get_text(separator=" ", strip=True)[:2000]
                        all_content.append("--- PAGE: " + current + " --- " + str(title) + " " + text)
                        for a in soup.find_all("a", href=True):
                            link = urljoin(current, a["href"])
                            if urlparse(link).netloc == domain and link not in visited:
                                to_visit.append(link)
                    except Exception as e:
                        all_content.append("--- PAGE: " + current + " --- failed to load: " + str(e))
            finally:
                driver.quit()
            crawl_dir = os.path.expanduser("~/claude-agent/crawled")
            os.makedirs(crawl_dir, exist_ok=True)
            crawl_file = os.path.join(crawl_dir, domain.replace(".", "_") + ".txt")
            full_text = " ".join(all_content)
            with open(crawl_file, "w") as f:
                f.write(full_text)
            summary_prompt = "I crawled " + url + " visited " + str(len(visited)) + " pages. Content: " + full_text[:4000] + " Summarize this website."
            self.title = "🤖"
            self.get_response(summary_prompt)
        except Exception as e:
            self.title = "🤖"
            self.show_response("Crawl error: " + str(e))

    @rumps.clicked("Crawl Website")
    def crawl_website(self, _):
        threading.Thread(target=self.do_crawl_website, daemon=True).start()

    @rumps.clicked("Search Profile")
    def insta_search(self, _):
        username = self.mac_input("Enter Instagram username to search:")
        if username:
            self.title = "🔍"
            threading.Thread(target=self._insta_search, args=(username,), daemon=True).start()

    def _insta_search(self, username):
        instagram_tools._client = None
        clean = username.split(",")[0].strip()
        result = instagram_tools.search_profile(clean)
        self.title = "🤖"
        self.show_response(result)

    @rumps.clicked("Download Reel")
    def insta_download(self, _):
        url = self.mac_input("Enter reel URL:")
        if url:
            self.title = "⬇️"
            threading.Thread(target=self._insta_download, args=(url,), daemon=True).start()

    def _insta_download(self, url):
        result = instagram_tools.download_reel(url)
        self.title = "🤖"
        self.show_response(result)

    @rumps.clicked("Post Reel")
    def insta_post(self, _):
        path = self.mac_pick_file("Select video file to post:")
        if not path:
            return
        caption = self.mac_input("Enter caption:")
        if caption is None:
            return
        self.title = "📤"
        threading.Thread(target=self._insta_post, args=(path, caption), daemon=True).start()

    @rumps.clicked("Download User Reels")
    def insta_download_user(self, _):
        username = self.mac_input("Enter Instagram username:")
        if not username:
            return
        count = self.mac_input("How many reels to download? (default 10)")
        count = int(count) if count and count.isdigit() else 10
        self.title = "⬇️"
        threading.Thread(target=self._insta_download_user, args=(username.split(",")[0].strip(), count), daemon=True).start()

    def _insta_download_user(self, username, count):
        self.show_response(f"Downloading {count} reels from {username}...")
        result = instagram_tools.download_reels_from_user(username, count)
        self.title = "🤖"
        self.show_response(result)

    def _insta_post(self, path, caption):
        result = instagram_tools.post_reel(path, caption)
        self.title = "🤖"
        self.show_response(result)

    @rumps.clicked("Talk to Me")
    def talk_to_me(self, _):
        threading.Thread(target=self._start_mood_chat, daemon=True).start()

    @rumps.clicked("Mood Chat")
    def mood_chat(self, _):
        threading.Thread(target=self._start_mood_chat, daemon=True).start()

    def _start_mood_chat(self):
        subprocess.Popen(["python3", os.path.expanduser("~/claude-agent/mood_chat.py")])

    @rumps.clicked("Video Editor")
    def video_editor_click(self, _):
        threading.Thread(target=self._open_video_editor, daemon=True).start()

    def _open_video_editor(self):
        def r(n1, t1, n2=None, t2=None):
            left = f"{n1:>2}. {t1}"
            if n2 is None:
                return left
            return f"{left}\t{n2:>2}. {t2}"
        menu = "\n".join([
            "AI VIDEO EDITOR  -  Enter a number",
            "",
            "[ BASIC EDITING ]",
            r(1,  "Add Captions",          2,  "Auto Captions (AI)"),
            r(3,  "Trim Video",            4,  "Merge Videos"),
            r(5,  "Add Background Music",  6,  "Extract Audio"),
            r(7,  "Change Speed",          8,  "Fade In / Out"),
            r(9,  "Add Intro / Outro",     10, "Resize for Platform"),
            "",
            "[ BACKGROUND ]",
            r(11, "Remove Background",     12, "Change Background (AI)"),
            r(13, "Chroma Key",            14, "Portrait Blur"),
            "",
            "[ FACE ]",
            r(15, "Face Blur",             16, "Face Pixelate"),
            r(17, "Face Swap (local)",     18, "Install Face Swap GAN"),
            r(19, "Enhance Faces",         20, "Setup Face Enhancer"),
            r(21, "Install GFPGAN"),
            "",
            "[ STYLING ]",
            r(22, "Color Grade",           23, "Add Watermark"),
            r(24, "Lower Third",           25, "Picture-in-Picture"),
            r(26, "SRT Subtitles",         27, "Add Transition"),
            r(28, "Zoom Effect"),
            "",
            "[ AI CREATION ]",
            r(29, "Ken Burns",             30, "AI Voiceover"),
            r(31, "Script to Video",       32, "AI Avatar"),
            "",
            "[ UTILITIES ]",
            r(33, "Slideshow",             34, "Extract Frames"),
            r(35, "Stabilize Video"),
            "",
            "[ WORKFLOWS ]",
            r(36, "Record Workflow",       37, "Run Saved Workflow"),
        ])
        choice = self.mac_input(menu)
        if not choice:
            return
        try:
            c = int(choice.strip())
        except ValueError:
            self.show_response("❌ Please enter a number.")
            return

        self.title = "🎬"
        if self._recording:
            self._begin_step(c)
        try:
            # ── BASIC EDITING ──────────────────────────────────────────
            if c == 1:
                video = self._ask_video("Enter video path:")
                if not video: return
                text = self.mac_input("Enter caption text:")
                if text:
                    result = video_editor.add_captions(video, text.strip())
                    self.show_response(result)
            elif c == 2:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.auto_captions(video)
                    self.show_response(result)
            elif c == 3:
                video = self._ask_video("Enter video path:")
                if not video: return
                start = self.mac_input("Start time in seconds:")
                end = self.mac_input("End time in seconds:")
                if start and end:
                    result = video_editor.trim_video(video, float(start), float(end))
                    self.show_response(result)
            elif c == 4:
                v1 = self._ask_video("Enter FIRST video path:")
                if not v1: return
                v2 = self._ask_video("Enter SECOND video path:")
                if v2:
                    result = video_editor.merge_videos([v1, v2])
                    self.show_response(result)
            elif c == 5:
                video = self._ask_video("Select video:")
                if not video: return
                music = self._ask_audio("Select music file:")
                if music:
                    vol = self.mac_input("Volume 0-1 (default 0.3):")
                    v = float(vol) if vol and vol.replace('.','').isdigit() else 0.3
                    result = video_editor.add_background_music(video, music, volume=v)
                    self.show_response(result)
            elif c == 6:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.extract_audio(video)
                    self.show_response(result)
            elif c == 7:
                video = self._ask_video("Enter video path:")
                if not video: return
                speed = self.mac_input("Speed multiplier (e.g. 1.5=faster, 0.5=slower):")
                if speed:
                    result = video_editor.change_speed(video, float(speed))
                    self.show_response(result)
            elif c == 8:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.add_fade(video)
                    self.show_response(result)
            elif c == 9:
                video = self._ask_video("Enter video path:")
                if not video: return
                intro = self.mac_input("Intro text (leave blank to skip):")
                outro = self.mac_input("Outro text (leave blank to skip):")
                result = video_editor.add_intro_outro(
                    video, intro.strip() if intro else "", outro.strip() if outro else "")
                self.show_response(result)
            elif c == 10:
                video = self._ask_video("Enter video path:")
                if not video: return
                plat = self.mac_input(
                    "Platform:\ninstagram / reels / tiktok / youtube / twitter / square")
                if plat:
                    result = video_editor.resize_for_platform(video, plat.strip().lower())
                    self.show_response(result)

            # ── BACKGROUND AI ─────────────────────────────────────────
            elif c == 11:
                img = self._ask_image("Select an image to remove background:")
                if img:
                    result = video_editor.remove_background_image(img)
                    self.show_response(result)
            elif c == 12:
                video = self._ask_video("Select video:")
                if not video: return
                bg_type = self.mac_input(
                    "New background type:\nfile / blur / color")
                if not bg_type: return
                bg_type = bg_type.strip().lower()
                if bg_type == 'blur':
                    bg_resolved = 'blur'
                elif bg_type == 'color':
                    hex_val = self.mac_input("Hex color (e.g. #1a1a2e):")
                    if not hex_val: return
                    bg_resolved = hex_val.strip()
                else:
                    bg_file = self.mac_pick_file("Select background image or video:")
                    if not bg_file: return
                    bg_resolved = bg_file
                result = video_editor.change_background(video, bg_resolved)
                self.show_response(result)
            elif c == 13:
                video = self._ask_video("Select green/blue screen video:")
                if not video: return
                bg = self.mac_pick_file("Select new background image:")
                if not bg: return
                color = self.mac_input("Screen color: green / blue (default green):")
                sc = color.strip().lower() if color and color.strip() else 'green'
                result = video_editor.chroma_key(video, bg, screen_color=sc)
                self.show_response(result)
            elif c == 14:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.background_blur(video)
                    self.show_response(result)

            # ── FACE FEATURES ─────────────────────────────────────────
            elif c == 15:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.face_blur(video)
                    self.show_response(result)
            elif c == 16:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.face_pixelate(video)
                    self.show_response(result)
            elif c == 17:
                video = self._ask_video("Select video:")
                if not video: return
                src_type = self.mac_input(
                    "Face source type:\nfile / @instagram / url")
                if not src_type: return
                src_type = src_type.strip().lower()
                if src_type == 'file':
                    face_src = self._ask_image("Select face photo:")
                else:
                    face_src = self.mac_input(
                        "@instagramusername  or  https://image-url:")
                if not face_src: return
                result = video_editor.face_swap(video, face_src.strip())
                self.show_response(result)

            # ── FACE FEATURES: GAN CHECK / INSTALL ────────────────────
            elif c == 18:
                status = video_editor.check_faceswap_engine()
                install = self.mac_input(
                    status + "\n\n─────\nType  install  to auto-install the GAN engine,\nor Cancel to close.")
                if install and install.strip().lower() == "install":
                    self.show_response(video_editor.install_faceswap_gan())
                else:
                    self.show_response(status)

            # ── FACE ENHANCEMENT ──────────────────────────────────────
            elif c == 19:
                video = self._ask_video("Enter video path to enhance\n"
                                        "(use on your face_swapped.mp4 for best results):")
                if not video: return
                weight = self.mac_input(
                    "Enhancement strength:\n0.3 = subtle  |  0.5 = balanced (default)  |  0.8 = strong")
                w = 0.5
                try:
                    if weight and weight.strip():
                        w = float(weight.strip())
                        w = max(0.1, min(1.0, w))
                except ValueError:
                    pass
                result = video_editor.enhance_faces(video, weight=w)
                self.show_response(result)

            # ── INSTALL CODEFORMER (primary face enhancer) ─────────────
            elif c == 20:
                self.show_response(
                    "🚀 Downloading CodeFormer ONNX …\n\n"
                    "This uses onnxruntime (already installed with insightface)\n"
                    "— NO new pip installs required.\n\n"
                    "This will:\n"
                    "  1. Verify onnxruntime is present\n"
                    "  2. Download codeformer.onnx (~180 MB)\n"
                    "  3. Run a load test\n\nStarting …")
                result = video_editor.install_codeformer()
                self.show_response(result)

            # ── INSTALL GFPGAN (legacy fallback) ───────────────────────
            elif c == 21:
                self.show_response(
                    "📥 Installing / patching GFPGAN face enhancer …\n"
                    "This patches the basicsr stub and verifies the full\n"
                    "GFPGAN pipeline is ready.\n\nStarting …")
                result = video_editor.install_face_enhancer()
                # Always patch the stub right after install to ensure scandir is present
                patch_result = video_editor.patch_basicsr_stub()
                self.show_response(result + '\n\n' + patch_result)

            # ── STYLING & EFFECTS ──────────────────────────────────────
            elif c == 22:
                video = self._ask_video("Enter video path:")
                if not video: return
                preset = self.mac_input(
                    "Color preset:\ncinematic / warm / cool / vintage / bw / "
                    "vivid / muted / neon / golden / teal_orange")
                if preset:
                    result = video_editor.color_grade(video, preset.strip().lower())
                    self.show_response(result)
            elif c == 23:
                video = self._ask_video("Enter video path:")
                if not video: return
                text = self.mac_input("Watermark text (e.g. @yourbrand):")
                if not text: return
                pos = self.mac_input(
                    "Position:\nbottom-right / bottom-left / top-right / top-left / center")
                p = pos.strip() if pos and pos.strip() in (
                    'bottom-right','bottom-left','top-right','top-left','center') else 'bottom-right'
                result = video_editor.add_watermark(video, text.strip(), position=p)
                self.show_response(result)
            elif c == 24:
                video = self._ask_video("Enter video path:")
                if not video: return
                name = self.mac_input("Name to display:")
                if not name: return
                title_text = self.mac_input("Title/role (leave blank to skip):")
                result = video_editor.add_lower_third(
                    video, name.strip(),
                    title=title_text.strip() if title_text else "")
                self.show_response(result)
            elif c == 25:
                main = self._ask_video("Enter MAIN video path:")
                if not main: return
                pip = self._ask_video("Enter PiP (overlay) video path:")
                if not pip: return
                pos = self.mac_input(
                    "PiP position:\nbottom-right / bottom-left / top-right / top-left")
                p = pos.strip() if pos and pos.strip() in (
                    'bottom-right','bottom-left','top-right','top-left') else 'bottom-right'
                result = video_editor.picture_in_picture(main, pip, position=p)
                self.show_response(result)
            elif c == 26:
                video = self._ask_video("Select video:")
                if not video: return
                srt = self.mac_pick_file("Select .srt subtitle file:",
                    file_types=["public.plain-text"])
                if srt:
                    result = video_editor.add_subtitles_srt(video, srt)
                    self.show_response(result)
            elif c == 27:
                v1 = self._ask_video("Enter FIRST video path:")
                if not v1: return
                v2 = self._ask_video("Enter SECOND video path:")
                if not v2: return
                tr = self.mac_input("Transition: crossfade / fade")
                t = tr.strip() if tr and tr.strip() in ('crossfade','fade') else 'crossfade'
                result = video_editor.add_transition(v1, v2, transition=t)
                self.show_response(result)
            elif c == 28:
                video = self._ask_video("Enter video path:")
                if not video: return
                z_end = self.mac_input("Zoom end multiplier (e.g. 1.5 = 50% zoom in):")
                ze = float(z_end) if z_end and z_end.replace('.','').isdigit() else 1.4
                result = video_editor.zoom_effect(video, zoom_end=ze)
                self.show_response(result)

            # ── AI CREATION ───────────────────────────────────────────
            elif c == 29:
                img = self._ask_image("Select image for Ken Burns effect:")
                if not img: return
                dur = self.mac_input("Duration in seconds (default 6):")
                d = float(dur) if dur and dur.replace('.','').isdigit() else 6.0
                result = video_editor.ken_burns(img, duration=d)
                self.show_response(result)
            elif c == 30:
                script = self.mac_input("Enter the script text for voiceover:")
                if not script: return
                voice = self.mac_input(
                    "Voice (Samantha/Alex/Victoria/Karen/Daniel - default Samantha):")
                v = voice.strip() if voice and voice.strip() else 'Samantha'
                result = video_editor.ai_voiceover(script.strip(), voice=v)
                self.show_response(result)
            elif c == 31:
                script = self.mac_input(
                    "Enter your full script:\n(It will be split into sentences, "
                    "each gets voice + caption + image)")
                if not script: return
                use_folder = self.mac_input(
                    "Use images folder? yes / no (default: gradient backgrounds):")
                folder = None
                if use_folder and use_folder.strip().lower() == 'yes':
                    folder = self.mac_pick_folder("Select images folder:")
                voice = self.mac_input(
                    "Voice (Samantha/Alex/Victoria - default Samantha):")
                v = voice.strip() if voice and voice.strip() else 'Samantha'
                result = video_editor.script_to_video(
                    script.strip(), images_folder=folder, voice=v)
                self.show_response(result)
            elif c == 32:
                script = self.mac_input("Enter the avatar's script (what it will say):")
                if not script: return
                photo = self._ask_image("Select your face photo:")
                if not photo: return
                name_text = self.mac_input("Name to show in lower third (blank to skip):")
                subtitle_text = self.mac_input("Subtitle/role text (blank to skip):")
                voice = self.mac_input(
                    "Voice (Samantha/Alex/Victoria/Karen - default Samantha):")
                v = voice.strip() if voice and voice.strip() else 'Samantha'
                plat = self.mac_input("Platform: portrait (default) / landscape / square")
                p = plat.strip() if plat and plat.strip() in ('portrait','landscape','square') else 'portrait'
                result = video_editor.create_ai_avatar(
                    script.strip(), photo, voice=v,
                    title=name_text.strip() if name_text else "",
                    subtitle=subtitle_text.strip() if subtitle_text else "",
                    platform=p)
                self.show_response(result)

            # ── UTILITIES ─────────────────────────────────────────────
            elif c == 33:
                folder = self.mac_pick_folder("Select images folder:")
                if not folder: return
                dur = self.mac_input("Seconds per image (default 3):")
                d = float(dur) if dur and dur.replace('.','').isdigit() else 3.0
                narr = self._ask_audio("Select narration audio (Cancel to skip):")
                result = video_editor.images_to_video(
                    folder, duration_per_image=d, narration_path=narr)
                self.show_response(result)
            elif c == 34:
                video = self._ask_video("Enter video path:")
                if not video: return
                every = self.mac_input("Extract a frame every N seconds (default 1):")
                n = float(every) if every and every.replace('.','').isdigit() else 1.0
                result = video_editor.extract_frames(video, every_n_seconds=n)
                self.show_response(result)
            elif c == 35:
                video = self._ask_video("Enter video path:")
                if video:
                    result = video_editor.stabilize_video(video)
                    self.show_response(result)

            # ── WORKFLOW RECORDING ─────────────────────────────────────
            elif c == 36:
                if not self._recording:
                    self._recording     = True
                    self._session_steps = []
                    self.show_response(
                        "Recording started.\n\n"
                        "Go through your workflow step by step.\n"
                        "Every action will be saved automatically.\n\n"
                        "When done, come back and choose option 36 again to stop and name it.")
                else:
                    self._recording = False
                    if not self._session_steps:
                        self.show_response("No steps recorded.")
                    else:
                        steps_summary = "\n".join(
                            f"  {i+1}. {s['name']}" for i, s in enumerate(self._session_steps))
                        wf_name = self.mac_input(
                            f"Recorded {len(self._session_steps)} steps:\n{steps_summary}\n\nName this workflow:")
                        if wf_name and wf_name.strip():
                            wf = self._load_workflows()
                            wf[wf_name.strip()] = {
                                'steps': self._session_steps,
                                'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            }
                            self._save_workflows(wf)
                            self.show_response(
                                f"Workflow '{wf_name.strip()}' saved with "
                                f"{len(self._session_steps)} steps.\n\n"
                                "Run it any time via option 37.")
                        else:
                            self.show_response("Workflow discarded (no name given).")
                    self._session_steps = []

            elif c == 37:
                wf = self._load_workflows()
                if not wf:
                    self.show_response("No saved workflows yet.\nRecord one first with option 36.")
                else:
                    names   = list(wf.keys())
                    listing = "\n".join(
                        f"  {i+1}. {n}  ({len(wf[n]['steps'])} steps, saved {wf[n].get('created','')})"
                        for i, n in enumerate(names))
                    choice2 = self.mac_input(
                        f"Saved workflows:\n{listing}\n\nEnter number to run:")
                    try:
                        idx = int(choice2.strip()) - 1
                        name = names[idx]
                    except (ValueError, TypeError, IndexError):
                        self.show_response("Invalid selection.")
                        return
                    self._run_saved_workflow(name)
                    self.show_response(f"Workflow '{name}' complete.")

            else:
                self.show_response(f"❌ Unknown option: {c}. Enter 1–37.")
        except Exception as e:
            self.show_response(f"Video error: {str(e)}")
        finally:
            self.title = "🤖"

    def _dispatch_video_option(self, c):
        """Re-run a single video editor option by number (used during workflow replay)."""
        # Delegates to the same elif chain inside _open_video_editor by re-entering
        # it with a fake choice. We do this by temporarily monkeypatching mac_input
        # so the menu dialog returns the option number string.
        _orig = self.mac_input
        called = [False]
        def _fake_input(prompt):
            if not called[0]:
                called[0] = True
                return str(c)
            return _orig(prompt)
        self.mac_input = _fake_input
        try:
            self._open_video_editor()
        finally:
            self.mac_input = _orig

    def _ask_video(self, prompt="Select a video file:"):
        """Open native file picker for a video file."""
        path = self.mac_pick_file(prompt,
            file_types=["public.movie", "public.mpeg-4",
                        "com.apple.quicktime-movie", "public.avi"],
            _is_video=True)
        if not path:
            return None
        return video_editor.resolve_path(path)

    def _ask_image(self, prompt="Select an image:"):
        """Open native file picker for an image file."""
        path = self.mac_pick_file(prompt,
            file_types=["public.image", "public.jpeg",
                        "public.png", "public.tiff"])
        return path if path else None

    def _ask_audio(self, prompt="Select an audio file:"):
        """Open native file picker for an audio file."""
        path = self.mac_pick_file(prompt,
            file_types=["public.audio", "public.mp3",
                        "com.apple.m4a-audio", "public.aiff-audio"])
        return path if path else None

    @rumps.clicked("Stop Speaking")
    def stop_speaking_click(self, _):
        self.stop_speaking()

    @rumps.clicked("Voice Output: ON")
    def toggle_voice(self, sender):
        self.voice_output = not self.voice_output
        sender.title = f"Voice Output: {'ON' if self.voice_output else 'OFF'}"

    @rumps.clicked("Clear History")
    def clear_history(self, _):
        self.history = []
        self.save_history()
        self.show_response("History cleared!")

if __name__ == "__main__":
    ClaudeAgent().run()
