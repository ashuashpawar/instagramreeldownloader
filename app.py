"""
Flask Web UI — Instagram Reels Downloader
==========================================
Cloud-ready version: session file loaded from INSTAGRAM_SESSION env var (base64).
Downloads saved to /tmp/reels_downloads/ and served via /files/<name>.

Install:
    pip install -r requirements.txt

Run locally:
    python3 app.py
    AGENT_PASSWORD=mypass python3 app.py

Run on cloud (Railway / Render):
    Set env vars AGENT_PASSWORD and INSTAGRAM_SESSION (see README).
    Start command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4
"""

import base64
import functools
import json
import mimetypes
import os
import queue
import secrets
import sys
import tempfile
import threading
import time

from flask import (Flask, Response, redirect, render_template,
                   request, send_file, session, stream_with_context, url_for)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ── Password ──────────────────────────────────────────────────────────────────
PASSWORD = os.environ.get("AGENT_PASSWORD", "reels2025")

# ── Download directory ────────────────────────────────────────────────────────
DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "reels_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ── Instagram session file ────────────────────────────────────────────────────
# On cloud: set INSTAGRAM_SESSION env var to base64-encoded contents of your .json session file.
# Locally: falls back to the hardcoded path in instagram_tools.py.
_session_file_path = None

def get_session_file() -> str:
    """Return path to a valid instagrapi session JSON file."""
    global _session_file_path

    # Already written this session
    if _session_file_path and os.path.exists(_session_file_path):
        return _session_file_path

    b64 = os.environ.get("INSTAGRAM_SESSION")
    if b64:
        data = base64.b64decode(b64)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _session_file_path = path
        return path

    # Fallback: use local instagram_tools path
    try:
        AGENT_DIR = os.path.expanduser("~/claude-agent")
        if AGENT_DIR not in sys.path:
            sys.path.insert(0, AGENT_DIR)
        import instagram_tools
        return instagram_tools.SESSION_FILE
    except Exception:
        raise RuntimeError(
            "No Instagram session found. Set the INSTAGRAM_SESSION env var "
            "(base64-encoded contents of your session .json file)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == PASSWORD:
            session["authenticated"] = True
            session.permanent = False
            return redirect(request.args.get("next") or url_for("index"))
        error = "Incorrect password."
        time.sleep(1)
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────────────────────
# FILE SERVING  (downloaded reels)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/files/<path:filename>")
@login_required
def serve_file(filename):
    """Serve a downloaded reel so the user can save it from their browser."""
    safe = os.path.basename(filename)          # strip any path traversal
    filepath = os.path.join(DOWNLOAD_DIR, safe)
    if not os.path.exists(filepath):
        return "File not found", 404
    mime, _ = mimetypes.guess_type(filepath)
    return send_file(filepath, mimetype=mime or "video/mp4", as_attachment=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def download_reels(username: str, count: int):
    try:
        from instagrapi import Client
    except ImportError as e:
        yield f"❌ Missing dependency: {e}\n"
        yield "   Run: pip install instagrapi\n"
        return

    yield f"📍 Profile  : @{username}\n"
    yield f"📦 Requested: {count} reel(s)\n"
    yield "─" * 48 + "\n"

    try:
        session_file = get_session_file()
    except RuntimeError as e:
        yield f"❌ {e}\n"
        return

    try:
        cl = Client()
        cl.load_settings(session_file)
        cl.request_timeout = 30
        time.sleep(1)

        yield f"🔍 Looking up @{username}…\n"
        user = cl.user_info_by_username(username)
        yield f"👤 {user.full_name} (@{user.username})  •  {user.follower_count:,} followers\n"
        yield f"📋 Fetching reel list…\n"

        medias = cl.user_clips(user.pk, amount=count)
        if not medias:
            yield f"⚠️  No reels found for @{username}\n"
            return

        yield f"   Found {len(medias)} reel(s)\n\n"

        downloaded = []

        for i, media in enumerate(medias):
            yield f"⬇️  [{i+1}/{len(medias)}] {media.pk}…\n"
            path = cl.video_download(media.pk, DOWNLOAD_DIR)
            fname = os.path.basename(str(path))
            downloaded.append(fname)
            # Special marker — UI turns this into a clickable download button
            yield f"__FILE__{fname}\n"
            time.sleep(2)

        yield "\n" + "─" * 48 + "\n"
        yield f"✅ Done — {len(downloaded)} reel(s) ready to download above\n"

    except Exception as e:
        yield f"\n❌ {e}\n"


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _stream(username: str, count: int):
    q      = queue.Queue()
    errors = []

    def worker():
        try:
            for chunk in download_reels(username, count):
                q.put(chunk)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            if errors:
                yield f"data: {json.dumps({'error': errors[0]})}\n\n"
            yield "data: [DONE]\n\n"
            break
        yield f"data: {json.dumps({'text': item})}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/download", methods=["POST"])
@login_required
def download():
    body     = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip().lstrip("@")
    count    = int(body.get("count") or 5)
    count    = max(1, min(count, 50))

    if not username:
        def _err():
            yield 'data: {"error": "Please enter an Instagram username."}\n\n'
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    return Response(
        stream_with_context(_stream(username, count)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Instagram Reels Downloader")
    print(f"  → http://localhost:{port}")
    print(f"  → Password : {PASSWORD}")
    print(f"  → Save dir : {DOWNLOAD_DIR}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
