"""
Flask Web UI — Instagram Reels Downloader
==========================================
Cloud-ready. Supports download by profile or direct reel URL.
Session loaded from INSTAGRAM_SESSION env var (base64).
Downloads saved to /tmp/reels_downloads/ and served via /files/<name>.

Start command:
    pip3 install -r requirements.txt --break-system-packages && python3 app.py
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

PASSWORD    = os.environ.get("AGENT_PASSWORD", "reels2025")
DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "reels_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

_session_file_path = None


def get_session_file() -> str:
    global _session_file_path
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
    try:
        AGENT_DIR = os.path.expanduser("~/claude-agent")
        if AGENT_DIR not in sys.path:
            sys.path.insert(0, AGENT_DIR)
        import instagram_tools
        return instagram_tools.SESSION_FILE
    except Exception:
        raise RuntimeError(
            "No Instagram session found. Set the INSTAGRAM_SESSION env var."
        )


def make_client():
    from instagrapi import Client
    cl = Client()
    cl.load_settings(get_session_file())
    cl.request_timeout = 30
    return cl


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
# FILE SERVING
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/files/<path:filename>")
@login_required
def serve_file(filename):
    safe     = os.path.basename(filename)
    filepath = os.path.join(DOWNLOAD_DIR, safe)
    if not os.path.exists(filepath):
        return "File not found", 404
    mime, _ = mimetypes.guess_type(filepath)
    return send_file(
        filepath,
        mimetype=mime or "video/mp4",
        as_attachment=True,
        download_name=safe,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD LOGIC — by profile
# ─────────────────────────────────────────────────────────────────────────────

def download_by_profile(username: str, count: int):
    try:
        from instagrapi import Client  # noqa — triggers ImportError early
    except ImportError as e:
        yield f"❌ Missing dependency: {e}\n"
        return

    yield f"📍 Profile  : @{username}\n"
    yield f"📦 Requested: {count} reel(s)\n"
    yield "─" * 48 + "\n"

    try:
        cl = make_client()
        time.sleep(0.5)

        yield f"🔍 Looking up @{username}…\n"
        user   = cl.user_info_by_username(username)
        yield f"👤 {user.full_name} (@{user.username})  •  {user.follower_count:,} followers\n"
        yield  "📋 Fetching reel list…\n"

        medias = cl.user_clips(user.pk, amount=count)
        if not medias:
            yield f"⚠️  No reels found for @{username}\n"
            return

        yield f"   Found {len(medias)} reel(s)\n\n"

        for i, media in enumerate(medias):
            yield f"⬇️  [{i+1}/{len(medias)}] {media.pk}…\n"
            path  = cl.video_download(media.pk, DOWNLOAD_DIR)
            fname = os.path.basename(str(path))
            yield f"__FILE__{fname}\n"
            time.sleep(0.5)   # reduced from 2 s → faster

        yield "\n" + "─" * 48 + "\n"
        yield f"✅ Done — {len(medias)} reel(s) ready above\n"

    except Exception as e:
        yield f"\n❌ {e}\n"


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD LOGIC — by direct URL
# ─────────────────────────────────────────────────────────────────────────────

def download_by_url(url: str):
    try:
        from instagrapi import Client  # noqa
    except ImportError as e:
        yield f"❌ Missing dependency: {e}\n"
        return

    yield f"🔗 URL: {url}\n"
    yield "─" * 48 + "\n"

    try:
        cl = make_client()

        yield "🔍 Fetching reel info…\n"
        media_pk = cl.media_pk_from_url(url)
        media    = cl.media_info(media_pk)
        yield f"👤 Reel by @{media.user.username}\n"

        yield "⬇️  Downloading…\n"
        path  = cl.video_download(media_pk, DOWNLOAD_DIR)
        fname = os.path.basename(str(path))
        yield f"__FILE__{fname}\n"

        yield "\n" + "─" * 48 + "\n"
        yield "✅ Done — reel ready above\n"

    except Exception as e:
        yield f"\n❌ {e}\n"


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _stream_gen(gen):
    q      = queue.Queue()
    errors = []

    def worker():
        try:
            for chunk in gen:
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


def sse_response(gen):
    return Response(
        stream_with_context(_stream_gen(gen)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    url      = (body.get("url")      or "").strip()
    username = (body.get("username") or "").strip().lstrip("@")
    count    = max(1, min(int(body.get("count") or 5), 50))

    if url:
        return sse_response(download_by_url(url))

    if username:
        return sse_response(download_by_profile(username, count))

    def _err():
        yield 'data: {"error": "Please enter a username or paste a reel URL."}\n\n'
        yield "data: [DONE]\n\n"
    return Response(stream_with_context(_err()), mimetype="text/event-stream")


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
