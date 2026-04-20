"""
AI Video Editor v2.0 — Comprehensive Video Production Suite
============================================================
Features:
  • Captions / Auto-captions (Whisper) — PIL-based, NO ImageMagick needed
  • Background removal & replacement (rembg AI)
  • Chroma key / green-screen removal
  • Background blur (portrait mode)
  • Color grading (cinematic, warm, cool, vintage, B&W, vivid, neon…)
  • Face blur / pixelate (OpenCV)
  • Watermark, lower-thirds, picture-in-picture
  • Ken Burns effect (pan & zoom on photos)
  • SRT subtitle support
  • AI voiceover (macOS TTS / OpenAI TTS)
  • Script-to-video pipeline (script → voice → captions → video)
  • AI Avatar video (photo + voice + captions)
  • Zoom effect, transitions, stabilization
  • Slideshow from image folder
  • Extract frames, merge, trim, speed change, fade, resize for platform
  • Add background music

Install dependencies:
  brew install ffmpeg imagemagick   (imagemagick only needed for legacy TextClip)
  pip3 install rembg opencv-python openai-whisper --break-system-packages
"""

import os
import re
import json
import time
import sys
import subprocess
import numpy as np
import textwrap
from datetime import datetime
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, VideoClip, ColorClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
    vfx, afx
)
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT = os.path.expanduser("~/Downloads/edited")
os.makedirs(OUTPUT, exist_ok=True)

PLATFORM_SIZES = {
    "instagram": (1080, 1920),
    "reels":     (1080, 1920),
    "tiktok":    (1080, 1920),
    "youtube":   (1920, 1080),
    "twitter":   (1280, 720),
    "square":    (1080, 1080),
    "landscape": (1920, 1080),
    "portrait":  (1080, 1920),
}

COLOR_PRESETS = {
    "cinematic": dict(contrast=1.2, saturation=0.85, brightness=0.95,
                      shadow=(10, 15, 30), highlight=(255, 245, 235)),
    "warm":      dict(contrast=1.1, saturation=1.1, brightness=1.05,
                      shadow=(20, 10, 0), highlight=(255, 250, 230)),
    "cool":      dict(contrast=1.1, saturation=0.9, brightness=1.0,
                      shadow=(0, 10, 20), highlight=(230, 240, 255)),
    "vintage":   dict(contrast=0.9, saturation=0.7, brightness=0.95,
                      shadow=(30, 20, 10), highlight=(255, 240, 200)),
    "bw":        dict(bw=True),
    "vivid":     dict(contrast=1.3, saturation=1.4, brightness=1.0),
    "muted":     dict(contrast=0.9, saturation=0.6, brightness=1.05),
    "neon":      dict(contrast=1.4, saturation=1.8, brightness=0.9),
    "golden":    dict(contrast=1.15, saturation=1.2, brightness=1.0,
                      shadow=(30, 20, 0), highlight=(255, 250, 200)),
    "teal_orange": dict(contrast=1.25, saturation=1.1, brightness=0.97,
                        shadow=(0, 30, 40), highlight=(255, 230, 180)),
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def resolve_path(path):
    """
    Resolve a video/image path to an existing file.

    Handles:
    • Embedded newlines / carriage returns in pasted paths
    • "Downloaded to: /path/file.mp4" prefix from download responses
    • Bare filename  →  searches ~/Downloads automatically
    • Partial name   →  fuzzy-matches files in ~/Downloads
    """
    if not path:
        return path

    # ── 1. De-duplicate repeated paths (osascript sometimes triples the value)
    # Split on newlines first; take the FIRST non-empty line only.
    lines = [l.strip() for l in path.splitlines() if l.strip()]
    path = lines[0] if lines else path.strip()

    # ── 2. Scrub residual whitespace noise ──────────────────────────────
    path = path.replace('\r', '').replace('\t', '').strip()

    # ── 2. Strip common prefixes added by the agent's own responses ─────
    for prefix in ("Downloaded to: ", "Saved to ", "✅ Saved: ", "✅ Saved to "):
        if path.startswith(prefix):
            path = path[len(prefix):].strip()

    # ── 3. Try the path exactly as given ───────────────────────────────
    if os.path.exists(path):
        return path

    # ── 4. Try inside ~/Downloads (user gave just the filename) ─────────
    downloads = os.path.expanduser("~/Downloads")
    basename = os.path.basename(path)          # handles "just-name.mp4" or full path
    full = os.path.join(downloads, basename)
    if os.path.exists(full):
        return full

    # ── 5. Fuzzy match inside ~/Downloads ───────────────────────────────
    exts = ('.mp4', '.mov', '.m4v', '.avi', '.mkv', '.jpg', '.jpeg', '.png', '.webp')
    name_lower = basename.lower()
    try:
        matches = [
            f for f in os.listdir(downloads)
            if name_lower in f.lower() and f.lower().endswith(exts)
        ]
        if matches:
            return os.path.join(downloads, sorted(matches)[0])
    except OSError:
        pass

    # ── 6. Return cleaned path (let the caller show a helpful error) ─────
    return path


def verify_video_path(path):
    """
    Raise a clear ValueError if the path doesn't point to a real file.
    Call this at the top of every function that opens a video.
    """
    if not os.path.isfile(path):
        # Try to give a helpful hint
        downloads = os.path.expanduser("~/Downloads")
        basename = os.path.basename(path)
        hint = ""
        try:
            close = [f for f in os.listdir(downloads) if basename[:10] in f]
            if close:
                hint = f"\nDid you mean: {close[0]}  ?"
        except OSError:
            pass
        raise FileNotFoundError(
            f"❌ File not found: {path}{hint}\n"
            "Tip: you can enter just the filename (e.g. myvideo.mp4) "
            "and it will be found automatically in ~/Downloads."
        )

def out_path(name):
    """
    Return a unique path inside OUTPUT — never overwrites an existing file.
    Temp files (names starting with _tmp_) are exempt and returned as-is.
    For real outputs: if the file exists, append _2, _3, … before the extension.
    """
    # Temp files are always overwritten (they're intermediate, cleaned up later)
    base, ext = os.path.splitext(name)
    if base.startswith('_tmp_'):
        return os.path.join(OUTPUT, name)

    candidate = os.path.join(OUTPUT, name)
    if not os.path.exists(candidate):
        return candidate

    # File exists — find the next free slot
    counter = 2
    while True:
        candidate = os.path.join(OUTPUT, f"{base}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1

def done(path):
    subprocess.Popen(["open", OUTPUT])
    return f"✅ Saved: {path}"


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING SYSTEM  — video editor hooks into learnings.json & profile.json
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_LEARNINGS_PATH = os.path.join(_AGENT_DIR, "learnings.json")
_PROFILE_PATH   = os.path.join(_AGENT_DIR, "profile.json")

def log_video_action(action: str, details: dict = None,
                     success: bool = True, error: str = None):
    """
    Record a video editing action into learnings.json and profile.json.

    Parameters
    ----------
    action  : short label, e.g. "face_swap", "color_grade", "add_captions"
    details : dict of settings used (preset, platform, fontsize, …)
    success : True if the operation completed without error
    error   : error message string if success=False
    """
    details = details or {}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Build a human-readable task string ──────────────────────────────────
    detail_str = ", ".join(f"{k}={v}" for k, v in details.items() if v is not None)
    task_label = f"video_editor: {action}" + (f" ({detail_str})" if detail_str else "")

    # ── Append entry to learnings.json ──────────────────────────────────────
    try:
        if os.path.exists(_LEARNINGS_PATH):
            with open(_LEARNINGS_PATH, "r") as f:
                learnings = json.load(f)
        else:
            learnings = []

        entry = {
            "date":        ts,
            "task":        task_label,
            "successes":   [f"Completed {action} with {detail_str}"] if success else [],
            "failures":    [error] if error else [],
            "total_steps": 1,
            "category":    "video_editor",
            "settings":    details,
        }
        learnings.append(entry)

        with open(_LEARNINGS_PATH, "w") as f:
            json.dump(learnings, f, indent=2)
    except Exception as e:
        print(f"[learning] Could not write learnings.json: {e}")

    # ── Update profile.json ─────────────────────────────────────────────────
    try:
        if os.path.exists(_PROFILE_PATH):
            with open(_PROFILE_PATH, "r") as f:
                profile = json.load(f)
        else:
            profile = {}

        # Ensure video-editing sub-sections exist
        profile.setdefault("video_editing", {
            "common_actions":   [],
            "success_patterns": [],
            "failure_patterns": [],
            "preferred_settings": {},
        })
        ve = profile["video_editing"]

        if success:
            # Track common actions (keep last 30 unique)
            if task_label not in ve["common_actions"]:
                ve["common_actions"].append(task_label)
                ve["common_actions"] = ve["common_actions"][-30:]

            # Track preferred settings per action
            if details:
                ve["preferred_settings"].setdefault(action, {})
                ve["preferred_settings"][action].update(details)

            # Track success pattern
            pat = f"{action}: {detail_str}" if detail_str else action
            if pat not in ve["success_patterns"]:
                ve["success_patterns"].append(pat)
                ve["success_patterns"] = ve["success_patterns"][-50:]
        else:
            # Track failure pattern
            pat = f"{action} failed: {error}" if error else f"{action} failed"
            if pat not in ve["failure_patterns"]:
                ve["failure_patterns"].append(pat)
                ve["failure_patterns"] = ve["failure_patterns"][-20:]

        with open(_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
    except Exception as e:
        print(f"[learning] Could not write profile.json: {e}")


def _timed_action(action: str, details: dict = None):
    """
    Context manager / decorator helper — returns (start_time, action, details).
    Use with _log_timed() below to auto-log duration.
    """
    return time.time()


def _log_timed(action: str, t0: float, details: dict = None,
               success: bool = True, error: str = None):
    """Log a video action and include how long it took."""
    details = details or {}
    elapsed = round(time.time() - t0, 1)
    details["duration_sec"] = elapsed
    log_video_action(action, details, success=success, error=error)


def _get_font(size):
    """Return best available PIL font for macOS."""
    candidates = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/HelveticaNeue.ttc',
        '/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/SFNSRounded.ttf',
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _text_size(draw, text, font):
    """Return (width, height) of text."""
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return len(text) * (font.size // 2), font.size

def _wrap_text(text, font, max_px):
    """Word-wrap text to fit within max_px width."""
    words = text.split()
    lines, current = [], []
    dummy = ImageDraw.Draw(Image.new('L', (1, 1)))
    for word in words:
        test = ' '.join(current + [word])
        try:
            bbox = dummy.textbbox((0, 0), test, font=font)
            w = bbox[2] - bbox[0]
        except Exception:
            w = len(test) * (font.size // 2)
        if w <= max_px:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
            current = [word]
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)

def make_text_overlay(text, clip_size, fontsize=50, color='white',
                      stroke_color='black', stroke_width=2,
                      bg_alpha=0.0, bg_color=None,
                      duration=None, position='bottom', padding=24):
    """
    Create a transparent text overlay clip using PIL — no ImageMagick needed.
    Returns a MoviePy ImageClip with alpha mask.
    """
    W, H = clip_size
    font = _get_font(fontsize)
    wrapped = _wrap_text(text, font, W - padding * 2)

    # Measure
    dummy_img = Image.new('RGBA', (W, H))
    dummy_draw = ImageDraw.Draw(dummy_img)
    tw, th = _text_size(dummy_draw, wrapped, font)

    # Position
    x = max(padding, (W - tw) // 2)
    if position == 'top':
        y = padding * 2
    elif position == 'center':
        y = (H - th) // 2
    else:  # bottom
        y = H - th - padding * 3

    # Draw on RGBA canvas
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent background box
    if bg_alpha > 0 or bg_color:
        box_fill = bg_color if bg_color else (0, 0, 0, int(bg_alpha * 255))
        try:
            draw.rounded_rectangle(
                [x - padding, y - padding // 2,
                 x + tw + padding, y + th + padding // 2],
                radius=10, fill=box_fill)
        except AttributeError:
            draw.rectangle(
                [x - padding, y - padding // 2,
                 x + tw + padding, y + th + padding // 2],
                fill=box_fill)

    # Stroke
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.multiline_text((x + dx, y + dy), wrapped, font=font,
                                    fill=stroke_color, spacing=6, align='center')
    # Main text
    draw.multiline_text((x, y), wrapped, font=font, fill=color,
                        spacing=6, align='center')

    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0

    clip = ImageClip(rgb, ismask=False)
    clip = clip.set_mask(ImageClip(alpha, ismask=True))
    if duration is not None:
        clip = clip.set_duration(duration)
    return clip

# ─────────────────────────────────────────────────────────────────────────────
# ① BASIC EDITING (FIXED — no ImageMagick)
# ─────────────────────────────────────────────────────────────────────────────

def add_captions(video_path, caption_text, fontsize=50,
                 color='white', position='bottom',
                 output_name="captioned.mp4"):
    """Add styled captions using PIL (ImageMagick-free)."""
    t0 = _timed_action("add_captions")
    try:
        verify_video_path(video_path)
        clip = VideoFileClip(video_path)
        txt = make_text_overlay(caption_text, clip.size, fontsize=fontsize,
                                color=color, stroke_color='black', stroke_width=2,
                                bg_alpha=0.5, duration=clip.duration, position=position)
        final = CompositeVideoClip([clip, txt])
        out = out_path(output_name)
        final.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("add_captions", t0, {"fontsize": fontsize, "color": color, "position": position})
        return done(out)
    except Exception as e:
        _log_timed("add_captions", t0, {"fontsize": fontsize}, success=False, error=str(e))
        raise

def trim_video(video_path, start_sec, end_sec, output_name="trimmed.mp4"):
    t0 = _timed_action("trim_video")
    try:
        verify_video_path(video_path)
        clip = VideoFileClip(video_path).subclip(start_sec, end_sec)
        out = out_path(output_name)
        clip.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("trim_video", t0, {"start_sec": start_sec, "end_sec": end_sec})
        return done(out)
    except Exception as e:
        _log_timed("trim_video", t0, {"start_sec": start_sec, "end_sec": end_sec}, success=False, error=str(e))
        raise

def merge_videos(video_paths, output_name="merged.mp4"):
    for p in video_paths:
        verify_video_path(p)
    clips = [VideoFileClip(p) for p in video_paths]
    target = clips[0].size
    clips = [c.resize(target) for c in clips]
    final = concatenate_videoclips(clips, method="compose")
    out = out_path(output_name)
    final.write_videofile(out, fps=clips[0].fps, logger=None)
    log_video_action("merge_videos", {"count": len(video_paths)})
    return done(out)

def add_background_music(video_path, music_path, volume=0.3,
                          output_name="with_music.mp4"):
    t0 = _timed_action("add_background_music")
    try:
        verify_video_path(video_path)
        clip = VideoFileClip(video_path)
        music = AudioFileClip(music_path).volumex(volume)
        music = music.subclip(0, clip.duration) if music.duration > clip.duration \
                else afx.audio_loop(music, duration=clip.duration)
        final_audio = CompositeAudioClip([clip.audio, music]) if clip.audio is not None else music
        final = clip.set_audio(final_audio)
        out = out_path(output_name)
        final.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("add_background_music", t0, {"volume": volume})
        return done(out)
    except Exception as e:
        _log_timed("add_background_music", t0, {"volume": volume}, success=False, error=str(e))
        raise

def extract_audio(video_path, output_name="audio.mp3"):
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    out = out_path(output_name)
    clip.audio.write_audiofile(out, logger=None)
    return done(out)

def resize_for_platform(video_path, platform="instagram", output_name=None):
    """Resize with letterbox/pillarbox padding to match platform aspect ratio."""
    t0 = _timed_action("resize_for_platform")
    try:
        w, h = PLATFORM_SIZES.get(platform.lower(), (1080, 1920))
        if not output_name:
            output_name = f"{platform}_{os.path.basename(video_path)}"
        verify_video_path(video_path)
        clip = VideoFileClip(video_path)
        clip_ratio = clip.w / clip.h
        target_ratio = w / h
        if clip_ratio > target_ratio:
            new_w, new_h = w, int(w / clip_ratio)
        else:
            new_w, new_h = int(h * clip_ratio), h
        resized = clip.resize((new_w, new_h))
        padded = resized.on_color(size=(w, h), color=(0, 0, 0), pos='center')
        out = out_path(output_name)
        padded.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("resize_for_platform", t0, {"platform": platform, "width": w, "height": h})
        return done(out)
    except Exception as e:
        _log_timed("resize_for_platform", t0, {"platform": platform}, success=False, error=str(e))
        raise

def add_intro_outro(video_path, intro_text="", outro_text="",
                    slide_duration=3, output_name="with_intro_outro.mp4"):
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    parts = []
    if intro_text:
        bg = ColorClip(clip.size, color=(0, 0, 0), duration=slide_duration)
        txt = make_text_overlay(intro_text, clip.size, fontsize=60, color='white',
                                duration=slide_duration, position='center')
        parts.append(CompositeVideoClip([bg, txt]).fadein(0.5).fadeout(0.5))
    parts.append(clip)
    if outro_text:
        bg = ColorClip(clip.size, color=(0, 0, 0), duration=slide_duration)
        txt = make_text_overlay(outro_text, clip.size, fontsize=60, color='white',
                                duration=slide_duration, position='center')
        parts.append(CompositeVideoClip([bg, txt]).fadein(0.5).fadeout(0.5))
    final = concatenate_videoclips(parts, method='compose')
    out = out_path(output_name)
    final.write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def change_speed(video_path, speed=1.5, output_name="speed_changed.mp4"):
    t0 = _timed_action("change_speed")
    try:
        verify_video_path(video_path)
        clip = VideoFileClip(video_path).fx(vfx.speedx, speed)
        out = out_path(output_name)
        clip.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("change_speed", t0, {"speed": speed})
        return done(out)
    except Exception as e:
        _log_timed("change_speed", t0, {"speed": speed}, success=False, error=str(e))
        raise

def add_fade(video_path, fade_duration=1, output_name="faded.mp4"):
    verify_video_path(video_path)
    clip = VideoFileClip(video_path).fadein(fade_duration).fadeout(fade_duration)
    out = out_path(output_name)
    clip.write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ② AUTO CAPTIONS WITH WHISPER
# ─────────────────────────────────────────────────────────────────────────────

def auto_captions(video_path, output_name="auto_captioned.mp4",
                  fontsize=40, model_size="base"):
    """
    Transcribe audio with Whisper and burn in timed captions (no ImageMagick).
    Install: pip3 install openai-whisper --break-system-packages
    """
    t0 = _timed_action("auto_captions")
    try:
        import whisper
    except ImportError:
        log_video_action("auto_captions", success=False, error="whisper not installed")
        return "❌ Install Whisper: pip3 install openai-whisper --break-system-packages"

    try:
        verify_video_path(video_path)
        clip = VideoFileClip(video_path)
        wav_path = out_path("_tmp_audio.wav")
        clip.audio.write_audiofile(wav_path, logger=None)

        model = whisper.load_model(model_size)
        result = model.transcribe(wav_path)
        os.remove(wav_path)

        txt_clips = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            tc = make_text_overlay(text, clip.size, fontsize=fontsize,
                                   color='white', stroke_color='black',
                                   stroke_width=2, bg_alpha=0.6, position='bottom')
            tc = tc.set_start(seg["start"]).set_end(min(seg["end"], clip.duration))
            txt_clips.append(tc)

        final = CompositeVideoClip([clip] + txt_clips)
        out = out_path(output_name)
        final.write_videofile(out, fps=clip.fps, logger=None)
        _log_timed("auto_captions", t0, {"model_size": model_size, "fontsize": fontsize,
                                          "segments": len(txt_clips)})
        return done(out)
    except Exception as e:
        _log_timed("auto_captions", t0, {"model_size": model_size}, success=False, error=str(e))
        raise

# ─────────────────────────────────────────────────────────────────────────────
# ③ BACKGROUND FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def remove_background_image(image_path, output_name="no_bg.png"):
    """Remove background from a single image using rembg AI.
    Install: pip3 install rembg --break-system-packages"""
    try:
        from rembg import remove
    except ImportError:
        return "❌ Install rembg: pip3 install rembg --break-system-packages"
    with open(image_path, 'rb') as f:
        result = remove(f.read())
    out = out_path(output_name)
    with open(out, 'wb') as f:
        f.write(result)
    return done(out)

def change_background(video_path, bg_source, output_name="bg_changed.mp4",
                      process_fps=None):
    """
    AI background removal + replacement for video.
    bg_source: image path  |  hex color '#rrggbb'  |  'blur'  |  video path
    process_fps: downsample to this fps for speed (None = use original fps)
    Install: pip3 install rembg --break-system-packages
    """
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        return "❌ Install rembg: pip3 install rembg --break-system-packages"

    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    fps = process_fps or clip.fps
    W, H = clip.size

    # Prepare background
    if bg_source.startswith('#') and len(bg_source) == 7:
        r, g, b = int(bg_source[1:3], 16), int(bg_source[3:5], 16), int(bg_source[5:7], 16)
        static_bg = Image.new('RGB', (W, H), (r, g, b))
        bg_type = 'color'
    elif os.path.exists(bg_source) and bg_source.lower().endswith(('.mp4', '.mov', '.m4v')):
        bg_clip_obj = VideoFileClip(bg_source).resize((W, H))
        bg_type = 'video'
        static_bg = None
    elif os.path.exists(bg_source):
        static_bg = Image.open(bg_source).convert('RGB').resize((W, H), Image.LANCZOS)
        bg_type = 'image'
    elif bg_source.lower() == 'blur':
        bg_type = 'blur'
        static_bg = None
    else:
        return f"❌ bg_source not found or invalid: {bg_source}"

    def process_frame(get_frame, t):
        frame = get_frame(t)
        pil_frame = Image.fromarray(frame).convert('RGBA')
        fg = rembg_remove(pil_frame)  # RGBA with transparent background

        if bg_type == 'blur':
            bg = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=25))
        elif bg_type == 'video':
            idx = min(int(t * bg_clip_obj.fps), int(bg_clip_obj.duration * bg_clip_obj.fps) - 1)
            bg = Image.fromarray(bg_clip_obj.get_frame(idx)).resize((W, H))
        else:
            bg = static_bg.copy()

        bg = bg.convert('RGBA')
        bg.paste(fg, (0, 0), fg)
        return np.array(bg.convert('RGB'))

    processed = clip.fl(process_frame)
    out = out_path(output_name)
    processed.write_videofile(out, fps=fps, logger=None)
    log_video_action("change_background", {"bg_type": bg_type, "process_fps": fps})
    return done(out)

def chroma_key(video_path, bg_source, screen_color='green',
               threshold=80, output_name="chroma_keyed.mp4"):
    """
    Remove green/blue screen and replace with new background.
    screen_color: 'green', 'blue', or (R,G,B) tuple
    bg_source: image path or '#rrggbb' hex color
    """
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    W, H = clip.size

    key_colors = {'green': (0, 255, 0), 'blue': (0, 0, 255), 'white': (255, 255, 255)}
    key_rgb = key_colors.get(screen_color, screen_color) \
              if isinstance(screen_color, str) else screen_color

    if bg_source.startswith('#'):
        r, g, b = int(bg_source[1:3], 16), int(bg_source[3:5], 16), int(bg_source[5:7], 16)
        bg_img = Image.new('RGB', (W, H), (r, g, b))
    elif os.path.exists(bg_source):
        bg_img = Image.open(bg_source).convert('RGB').resize((W, H), Image.LANCZOS)
    else:
        bg_img = Image.new('RGB', (W, H), (0, 0, 0))

    bg_arr = np.array(bg_img, dtype=np.float32)
    key_arr = np.array(key_rgb, dtype=np.float32)

    def remove_chroma(get_frame, t):
        frame = get_frame(t).astype(np.float32)
        dist = np.linalg.norm(frame - key_arr, axis=2)
        mask = np.clip(dist / threshold, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        m3 = np.stack([mask, mask, mask], axis=2)
        result = m3 * frame + (1 - m3) * bg_arr
        return np.clip(result, 0, 255).astype(np.uint8)

    processed = clip.fl(remove_chroma)
    out = out_path(output_name)
    processed.write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def background_blur(video_path, blur_radius=30, output_name="bg_blurred.mp4"):
    """Portrait-mode background blur (blurs everything outside center ellipse)."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)

    def blur_bg(get_frame, t):
        frame = get_frame(t)
        H, W = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (W // 2, H // 2), (W // 3, int(H * 0.45)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        m3 = np.stack([mask, mask, mask], axis=2)
        return (m3 * frame + (1 - m3) * blurred).astype(np.uint8)

    out = out_path(output_name)
    clip.fl(blur_bg).write_videofile(out, fps=clip.fps, logger=None)
    log_video_action("background_blur", {"blur_radius": blur_radius})
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ④ FACE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def check_faceswap_engine():
    """
    Report which face-swap engine is available and print install help if needed.
    Returns a human-readable status string.
    """
    lines = ["🎭 Face Swap Engine Status\n" + "─"*34]

    # ── InsightFace (GAN-based, production quality) ───────────────────
    try:
        import insightface          # noqa: F401
        import onnxruntime          # noqa: F401
        lines.append("✅  GAN engine ready  (InsightFace + inswapper_128)")
        lines.append("    Quality: Kling-level  •  Identity transfer  •  All angles")

        model_ok = os.path.exists(_INSWAPPER_PATH)
        if model_ok:
            size_mb = os.path.getsize(_INSWAPPER_PATH) // (1024*1024)
            lines.append(f"✅  inswapper_128.onnx downloaded  ({size_mb} MB)")
        else:
            lines.append("⚠️   inswapper_128 model NOT downloaded yet")
            lines.append("    It will download automatically (~260 MB) on first face swap run.")
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        lines.append(f"❌  GAN engine NOT installed  (missing: {missing})")
        lines.append("")
        lines.append("  👉 To install (run in your Mac Terminal):")
        lines.append("     pip3 install insightface onnxruntime --break-system-packages")
        lines.append("")
        lines.append("  After install, face swap will automatically use the GAN engine.")
        lines.append("  First run downloads models once (~460 MB total), then works offline.")

    # ── OpenCV fallback ───────────────────────────────────────────────
    lines.append("")
    lines.append("✅  OpenCV fallback always available  (tracked Poisson blend)")
    lines.append("    Quality: decent  •  No identity transfer  •  Face must face forward")

    return "\n".join(lines)


def install_faceswap_gan():
    """
    Attempt to install InsightFace + onnxruntime via pip3 and report result.
    Must be called with internet access.
    """
    print("⬇️  Installing InsightFace GAN engine …")
    result = subprocess.run(
        ["pip3", "install", "insightface", "onnxruntime", "--break-system-packages"],
        capture_output=False, text=True)
    if result.returncode == 0:
        return "✅ InsightFace installed! Restart the agent and run face swap — it will now use the GAN engine."
    else:
        return (
            "❌ pip3 install failed (possibly no internet in this environment).\n"
            "Run this command in your Mac Terminal manually:\n\n"
            "  pip3 install insightface onnxruntime --break-system-packages"
        )


_FACE_CASCADE = None  # cached — loaded once, not per-frame

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _FACE_CASCADE

def _detect_faces(gray_frame):
    return _get_face_cascade().detectMultiScale(gray_frame, 1.1, 4, minSize=(25, 25))

def face_blur(video_path, blur_strength=35, padding=20,
              output_name="face_blurred.mp4"):
    """Detect and blur all faces in video using OpenCV."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)

    def process(get_frame, t):
        frame = get_frame(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = _detect_faces(gray)
        result = frame.copy()
        for raw in faces:
            x, y, w, h = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            roi = result[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
        return result

    out = out_path(output_name)
    clip.fl(process).write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def face_pixelate(video_path, pixel_size=15, output_name="face_pixelated.mp4"):
    """Pixelate faces in video."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)

    def process(get_frame, t):
        frame = get_frame(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = _detect_faces(gray)
        result = frame.copy()
        for raw in faces:
            x, y, w, h = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
            roi = result[y:y + h, x:x + w]
            small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)))
            result[y:y + h, x:x + w] = cv2.resize(
                small, (w, h), interpolation=cv2.INTER_NEAREST)
        return result

    out = out_path(output_name)
    clip.fl(process).write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def _is_description_json(data):
    """Return True if this JSON is an AI visual-description of a person (not image data)."""
    desc_keys = {'subject', 'facial_features', 'hair', 'makeup', 'photography',
                 'face_shape', 'skin_tone', 'eyes', 'expression'}
    flat_keys = set()
    if isinstance(data, dict):
        flat_keys.update(data.keys())
        for v in data.values():
            if isinstance(v, dict):
                flat_keys.update(v.keys())
    return len(flat_keys & desc_keys) >= 2


def _description_json_to_prompt(data):
    """
    Convert an AI visual-description JSON into a detailed image generation prompt.
    Works with the format: {subject, facial_features, hair, makeup, outfit, photography}
    """
    parts = []

    subj = data.get('subject', {})
    face = data.get('facial_features', {})
    hair = data.get('hair', {})
    makeup = data.get('makeup', {})
    outfit = data.get('outfit', {})
    photo = data.get('photography', {})
    mood = data.get('mood', '')

    # Core subject
    gender = subj.get('gender', '')
    age = subj.get('apparent_age', '')
    pose = subj.get('pose', '')
    if gender or age:
        parts.append(f"Portrait photo of a {age} {gender}".strip())

    # Facial features
    face_shape = face.get('face_shape', '')
    skin = face.get('skin_tone', '')
    eyes = face.get('eyes', '')
    brows = face.get('eyebrows', '')
    lips = face.get('lips', '')
    expr = face.get('expression', '')
    facial_desc = ', '.join(filter(None, [
        f"{face_shape} face shape" if face_shape else '',
        f"{skin} skin tone" if skin else '',
        f"{eyes} eyes" if eyes else '',
        f"{brows} eyebrows" if brows else '',
        f"{lips} lips" if lips else '',
        expr
    ]))
    if facial_desc:
        parts.append(facial_desc)

    # Hair
    h_color = hair.get('color', '')
    h_len = hair.get('length', '')
    h_tex = hair.get('texture', '')
    h_style = hair.get('style', '')
    hair_desc = ' '.join(filter(None, [h_color, h_len, h_tex, h_style, 'hair']))
    if hair_desc.strip() != 'hair':
        parts.append(hair_desc)

    # Makeup
    mk_style = makeup.get('overall_style', '')
    mk_eyes = makeup.get('eyes', '')
    mk_lips = makeup.get('lips', '')
    if mk_style:
        parts.append(f"{mk_style} makeup")
    elif mk_eyes or mk_lips:
        parts.append(f"makeup: {mk_eyes}, {mk_lips} lips".strip(', '))

    # Outfit
    o_type = outfit.get('type', '')
    o_color = outfit.get('color', '')
    o_pattern = outfit.get('pattern', '')
    if o_type:
        parts.append(f"wearing {o_color} {o_pattern} {o_type}".strip())

    # Photography style
    shot = photo.get('shot_type', 'portrait')
    angle = photo.get('angle', '')
    bg = photo.get('background', '')
    lighting = photo.get('lighting', 'natural light')
    parts.append(f"{shot}, {angle}, {lighting}".strip(', '))
    if bg:
        parts.append(f"background: {bg}")

    if mood:
        parts.append(mood)

    # Quality boosters
    parts.append("photorealistic, high resolution, sharp focus on face, 4K")

    prompt = ', '.join(filter(None, parts))
    return prompt


def _generate_face_from_description(description_data, openai_api_key=None,
                                     output_name="_generated_face.png"):
    """
    Generate a face image from a description JSON using:
    1. OpenAI DALL-E 3 (if openai_api_key provided or OPENAI_API_KEY env var set)
    2. Stable Diffusion via Hugging Face Inference API (free, no key needed for basic use)
    3. Returns the saved image path
    """
    import urllib.request, urllib.error, tempfile, base64

    prompt = _description_json_to_prompt(description_data)
    tmp_path = out_path(output_name)

    # ── Option 1: OpenAI DALL-E 3 ──────────────────────────────────────
    key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if key:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "standard",
                "response_format": "b64_json"
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/images/generations",
                data=payload,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read())
            img_data = base64.b64decode(result['data'][0]['b64_json'])
            with open(tmp_path, 'wb') as f:
                f.write(img_data)
            return tmp_path, prompt

        except Exception as e:
            print(f"DALL-E failed: {e}, trying Hugging Face...")

    # ── Option 2: Stable Diffusion via Hugging Face (free) ─────────────
    try:
        import json as _json
        hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        hf_key = os.environ.get('HF_API_KEY', '')  # optional, works without for basic
        headers = {"Content-Type": "application/json"}
        if hf_key:
            headers["Authorization"] = f"Bearer {hf_key}"

        payload = _json.dumps({"inputs": prompt,
                               "parameters": {"width": 512, "height": 512}}).encode()
        req = urllib.request.Request(hf_url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            img_bytes = resp.read()
        with open(tmp_path, 'wb') as f:
            f.write(img_bytes)
        return tmp_path, prompt

    except Exception as e:
        raise RuntimeError(
            f"Could not generate face image automatically.\n\n"
            f"Generated prompt for manual use:\n{prompt}\n\n"
            f"To generate:\n"
            f"  • Set OPENAI_API_KEY in your environment and re-run, OR\n"
            f"  • Visit https://openai.com/dall-e-3 and paste the prompt above, OR\n"
            f"  • Use https://stablediffusionweb.com or Midjourney with the prompt\n"
            f"  • Then provide the generated image path for face swap.\n\n"
            f"Error: {e}")


def _resolve_face_image(face_source, openai_api_key=None):
    """
    Resolve any face source into a local image path.

    Accepted formats:
      1. Image path (.jpg/.png etc.)
      2. Instagram @username  — downloads their profile picture live
      3. JSON with image data keys:
           'image_path'      → path to local image
           'image_base64'    → base64-encoded image bytes
           'profile_pic_url' / 'image_url' / 'url'  → HTTP URL to download
      4. JSON with 'cookies'/'authorization_data' — Instagram session file
         (fetches the logged-in account's own profile picture)
      5. AI DESCRIPTION JSON  — has 'facial_features', 'subject', 'hair' etc.
         → Converts description to a prompt and generates image via DALL-E / SD
      6. Raw HTTP URL
      7. Inline JSON string  — e.g. {"facial_features": {...}, ...}

    Returns (resolved_image_path, temp_file_to_delete_or_None)
    """
    import json as _json
    import tempfile, base64, urllib.request

    # De-duplicate if osascript tripled the path (split on newlines, take first)
    lines = [l.strip() for l in face_source.splitlines() if l.strip()]
    src = lines[0] if lines else face_source.strip()

    # ── 1. Real image file ───────────────────────────────────────────────
    if os.path.isfile(src) and src.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')):
        return src, None

    # ── Helper: process a loaded dict ────────────────────────────────────
    def _process_dict(data, json_file_path=None):
        # a) AI visual-description JSON (the key case the user described)
        if _is_description_json(data):
            img_path, prompt = _generate_face_from_description(
                data, openai_api_key=openai_api_key)
            print(f"[face_swap] Generated face from description.\nPrompt used: {prompt}")
            return img_path, img_path  # temp file, caller cleans up

        # b) Explicit image path
        if 'image_path' in data and os.path.isfile(str(data['image_path'])):
            return data['image_path'], None

        # c) Base64 image
        if 'image_base64' in data:
            raw = base64.b64decode(data['image_base64'])
            tmp = tempfile.mktemp(suffix='.png')
            with open(tmp, 'wb') as f:
                f.write(raw)
            return tmp, tmp

        # d) URL inside JSON
        for key in ('profile_pic_url', 'profile_picture_url', 'avatar_url',
                    'image_url', 'photo_url', 'url'):
            if key in data and isinstance(data.get(key), str) \
                    and data[key].startswith('http'):
                tmp = tempfile.mktemp(suffix='.jpg')
                urllib.request.urlretrieve(data[key], tmp)
                return tmp, tmp

        # e) Instagram session file
        if json_file_path and ('cookies' in data or 'authorization_data' in data):
            return _fetch_insta_profile_pic_from_session(json_file_path)

        raise ValueError(
            "JSON found but no recognised keys.\n"
            "Expected: 'facial_features' (description), 'image_path', "
            "'image_base64', 'profile_pic_url', or Instagram session keys.")

    # ── 2. JSON file on disk ─────────────────────────────────────────────
    if os.path.isfile(src) and src.lower().endswith('.json'):
        with open(src, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        return _process_dict(data, json_file_path=src)

    # ── 3. Inline JSON string ────────────────────────────────────────────
    if src.startswith('{'):
        try:
            data = _json.loads(src)
            return _process_dict(data)
        except _json.JSONDecodeError:
            pass

    # ── 4. Instagram username ─────────────────────────────────────────────
    if src.startswith('@') or (
            not src.startswith('http') and not os.path.exists(src)
            and '.' not in os.path.basename(src)):
        username = src.lstrip('@').strip()
        return _fetch_insta_profile_pic(username)

    # ── 5. Raw HTTP URL ───────────────────────────────────────────────────
    if src.startswith('http'):
        tmp = tempfile.mktemp(suffix='.jpg')
        urllib.request.urlretrieve(src, tmp)
        return tmp, tmp

    raise ValueError(
        f"Cannot resolve face source: '{face_source}'\n"
        "Valid options:\n"
        "  • An image file path (.jpg, .png …)\n"
        "  • An AI description JSON file path (.json)\n"
        "  • An Instagram @username\n"
        "  • A direct image URL\n"
        "  • A JSON string pasted inline")


def _fetch_insta_profile_pic(username):
    """Download an Instagram profile picture by username using existing session."""
    import tempfile, urllib.request
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/claude-agent"))
        from instagrapi import Client
        SESSION_FILE = os.path.expanduser("~/claude-agent/instagram_tejalsonavne70.json")
        cl = Client()
        cl.load_settings(SESSION_FILE)
        user = cl.user_info_by_username(username)
        pic_url = str(user.profile_pic_url)
        tmp_path = tempfile.mktemp(suffix='.jpg')
        urllib.request.urlretrieve(pic_url, tmp_path)
        return tmp_path, tmp_path
    except Exception as e:
        raise RuntimeError(f"Could not fetch Instagram profile pic for @{username}: {e}")


def _fetch_insta_profile_pic_from_session(session_json_path):
    """
    The instagram_tejalsonavne70.json is a SESSION file (cookies, not photo).
    We use it to log in and then ask: whose face do you want?
    Returns (image_path, temp_path_or_None).
    """
    import tempfile, urllib.request
    try:
        from instagrapi import Client
        cl = Client()
        cl.load_settings(session_json_path)
        # Fetch the logged-in account's own profile pic
        user_id = cl.user_id
        user = cl.user_info(user_id)
        pic_url = str(user.profile_pic_url)
        tmp_path = tempfile.mktemp(suffix='.jpg')
        urllib.request.urlretrieve(pic_url, tmp_path)
        return tmp_path, tmp_path
    except Exception as e:
        raise RuntimeError(
            f"Could not use Instagram session to get profile pic: {e}\n"
            "Tip: pass @username instead, e.g. @tejalsonavne70")


def _largest_face(faces):
    """Return the largest face rect (by area) from detectMultiScale output."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: int(f[2]) * int(f[3]))


def _select_target_face(faces, rule, frame_w=0, frame_h=0):
    """
    Pick one InsightFace face object from a list using a selection rule.

    rule options:
      'largest' / '1'  – biggest bounding box (default)
      '2' / '3'        – 2nd or 3rd largest
      'left'           – leftmost face centre
      'right'          – rightmost face centre
      'top'            – highest face centre (smallest y)
      'bottom'         – lowest face centre
      'center'         – closest to the frame centre

    Falls back to largest if the rule can't be satisfied (e.g. '2' but only 1 face).
    """
    if not faces:
        return None

    by_size = sorted(faces,
                     key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                     reverse=True)

    # Integer index (from pick_hero_face) — direct lookup into size-sorted list
    if isinstance(rule, int):
        return by_size[rule] if rule < len(by_size) else by_size[0]

    rule = str(rule).strip().lower()

    if rule in ('largest', '1', ''):
        return by_size[0]
    if rule == '2':
        return by_size[1] if len(by_size) >= 2 else by_size[0]
    if rule == '3':
        return by_size[2] if len(by_size) >= 3 else by_size[0]

    # Position-based — use face centre coordinates
    cx_f = [(f.bbox[0] + f.bbox[2]) / 2 for f in faces]
    cy_f = [(f.bbox[1] + f.bbox[3]) / 2 for f in faces]

    if rule == 'left':
        return faces[int(np.argmin(cx_f))]
    if rule == 'right':
        return faces[int(np.argmax(cx_f))]
    if rule == 'top':
        return faces[int(np.argmin(cy_f))]
    if rule == 'bottom':
        return faces[int(np.argmax(cy_f))]
    if rule == 'center':
        cx, cy = frame_w / 2.0, frame_h / 2.0
        dists  = [abs(cx_f[i] - cx) + abs(cy_f[i] - cy) for i in range(len(faces))]
        return faces[int(np.argmin(dists))]

    # Unknown rule — fall back to largest
    return by_size[0]


def pick_hero_face(video_path):
    """
    Extract a frame, draw numbered boxes on every detected face,
    open the image in macOS Preview, then ask the user to type a number.

    Returns an integer index (0-based, sorted by face size descending)
    so face_swap knows which face to target.
    If only one face is found, or InsightFace is unavailable,
    returns 0 (largest) without showing any dialog.
    """
    import tempfile, subprocess as _sp

    # ── 1. Try to detect faces using InsightFace ──────────────────────
    try:
        from insightface.app import FaceAnalysis as _FA
        CPU = ['CPUExecutionProvider']
        model_name = 'buffalo_l' if os.path.isdir(
            os.path.expanduser('~/.insightface/models/buffalo_l')) else 'buffalo_sc'
        _app = _FA(name=model_name, providers=CPU)
        _app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        return 0   # InsightFace not installed — just use default

    # ── 2. Grab a frame ~2 s in (avoids black/fade-in frames) ─────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return 0

    faces = _app.get(frame)
    if len(faces) <= 1:
        return 0   # single face — no picker needed

    # Sort by area descending (face 1 = biggest)
    faces_sorted = sorted(faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True)

    # ── 3. Draw numbered boxes on a copy of the frame ─────────────────
    preview = frame.copy()
    colours = [(0,220,0),(0,120,255),(0,0,255),(255,150,0),(180,0,255)]
    for idx, face in enumerate(faces_sorted):
        x1,y1,x2,y2 = [int(v) for v in face.bbox]
        col = colours[idx % len(colours)]
        cv2.rectangle(preview, (x1,y1), (x2,y2), col, 3)
        label = str(idx + 1)
        fs = max(1.4, (y2-y1) / 120)
        lw = max(2, int(fs * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        # filled background chip so number is easy to read
        cv2.rectangle(preview, (x1, y1-th-12), (x1+tw+10, y1), col, -1)
        cv2.putText(preview, label, (x1+5, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), lw, cv2.LINE_AA)

    # ── 4. Save preview and open in macOS Preview ─────────────────────
    tmp_img = tempfile.mktemp(suffix='_face_pick.jpg')
    cv2.imwrite(tmp_img, preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _sp.Popen(['open', tmp_img])   # non-blocking — Preview opens alongside dialog

    # ── 5. Ask user to type a number ─────────────────────────────────
    n = len(faces_sorted)
    prompt = f"Look at the preview image.\nFaces are numbered 1–{n}.\n\nEnter the face number to swap (1 = biggest):"
    with open('/tmp/claude_prompt.txt', 'w') as _f:
        _f.write(prompt)
    script = '''
set filePath to "/tmp/claude_prompt.txt"
set fileRef to open for access POSIX file filePath
set msg to read fileRef
close access fileRef
set dlg to display dialog msg default answer "1" with title "Pick Hero Face" buttons {"Cancel", "OK"} default button "OK"
return text returned of dlg
'''
    res = _sp.run(['osascript', '-e', script], capture_output=True, text=True)
    try:
        chosen = int(res.stdout.strip()) - 1   # 0-based
        chosen = max(0, min(chosen, n - 1))
    except (ValueError, TypeError):
        chosen = 0

    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# ④-A  DEEP LEARNING FACE SWAP  (InsightFace + inswapper_128)
#       Install: pip3 install insightface onnxruntime --break-system-packages
#       First run auto-downloads models (~400 MB total).
#       This is the same stack used by Kling, roop, FaceFusion, etc.
# ─────────────────────────────────────────────────────────────────────────────

# ── Swap model candidates (tried in order, first success wins) ───────────────
# inswapper_128 was pulled from deepinsight HuggingFace due to licensing.
# These mirrors / alternatives are publicly accessible without auth.
_MODELS_DIR  = os.path.expanduser("~/.ai_faceswap/models")
_SWAP_MODELS = [
    # ① inswapper_128 community mirror (ezioruan) — best quality, ~554 MB
    (
        "inswapper_128.onnx",
        200_000_000,
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    ),
    # ② inswapper_128 another community mirror (Patil)
    (
        "inswapper_128.onnx",
        200_000_000,
        "https://huggingface.co/Patil/inswapper/resolve/main/inswapper_128.onnx",
    ),
    # ③ GHOST 256 — open-source GAN, good identity transfer, ~515 MB
    (
        "ghost_1_256.onnx",
        100_000_000,
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx",
    ),
    # ④ SimSwap 256 — CC-BY-NC, academic use, ~220 MB
    (
        "simswap_256.onnx",
        50_000_000,
        "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/simswap_256.onnx",
    ),
]


def _ensure_swap_model():
    """
    Return path to a working face-swap ONNX model, downloading if needed.
    Uses curl (macOS built-in) — reliable, shows progress, no auth issues.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)

    # Return first model that already exists on disk
    for fname, min_sz, _ in _SWAP_MODELS:
        p = os.path.join(_MODELS_DIR, fname)
        if os.path.exists(p) and os.path.getsize(p) > min_sz:
            return p

    # Download the first model that succeeds
    for fname, min_sz, url in _SWAP_MODELS:
        dest = os.path.join(_MODELS_DIR, fname)
        tmp  = dest + ".part"
        print(f"⬇️  Downloading {fname} …")
        try:
            result = subprocess.run([
                "curl", "-L", "-C", "-",
                "--retry", "3", "--retry-delay", "2",
                "--progress-bar",
                url, "-o", tmp
            ])
            if result.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > min_sz:
                os.replace(tmp, dest)
                print(f"✅ {fname} ready.")
                return dest
            # Bad download — clean up and try next
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"   ⚠️  Download incomplete, trying next model …")
        except Exception as e:
            print(f"   curl error: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    raise RuntimeError(
        "❌ Could not download any face swap model.\n"
        "Check your internet connection and try again."
    )


def _build_face_mask(face, W, H, feather=55):
    """
    Build a precise, feathered face mask using InsightFace landmarks.
    Falls back to ellipse if landmarks unavailable.
    Returns float32 mask (H, W) with values 0-1.
    """
    b = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0,b[0]), max(0,b[1]), min(W,b[2]), min(H,b[3])
    mask = np.zeros((H, W), dtype=np.float32)

    # Try 2D landmark convex hull (available with buffalo_l / buffalo_sc)
    pts = None
    for attr in ('landmark_2d_106', 'landmark_68_2d', 'kps'):
        lm = getattr(face, attr, None)
        if lm is not None and len(lm) >= 5:
            pts = lm.astype(np.int32)
            break

    if pts is not None:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)
        # Expand slightly so boundary pixels are covered
        k = max(3, (x2-x1)//8) | 1   # odd kernel
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    else:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        rx, ry = (x2-x1)//2, (y2-y1)//2
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    # Soft feathered edge
    ksize = feather | 1           # must be odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather // 3)
    return mask


def _reinhard_color_match(src_bgr, ref_bgr):
    """
    Reinhard per-channel mean/std color transfer.
    Matches the colour statistics of src to ref.
    Used to align swapped face skin tone to surrounding area.
    """
    out = src_bgr.astype(np.float32)
    ref = ref_bgr.astype(np.float32)
    for c in range(3):
        s_m = out[:,:,c].mean();  s_s = out[:,:,c].std()  + 1e-6
        r_m = ref[:,:,c].mean();  r_s = ref[:,:,c].std()  + 1e-6
        out[:,:,c] = (out[:,:,c] - s_m) * (r_s / s_s) + r_m
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_face(frame, x1, y1, x2, y2, strength=0.65, kps=None):
    """
    Multi-pass unsharp-mask sharpening on the swapped face region.
    • Pass 1 (sigma=0.8, strength=strength): fine-grained detail — pores, lips
    • Pass 2 (sigma=0.4, strength×0.8):       ultra-fine — restores 128px loss
    • Eye-region boost: extra sharpening on left/right eye areas using kps
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2].astype(np.float32)

    # Pass 1 — fine detail (tight sigma)
    blur1 = cv2.GaussianBlur(roi, (0, 0), 0.8)
    roi   = np.clip(roi + strength * (roi - blur1), 0, 255)

    # Pass 2 — ultra-fine (very tight sigma)
    blur2 = cv2.GaussianBlur(roi, (0, 0), 0.4)
    roi   = np.clip(roi + (strength * 0.8) * (roi - blur2), 0, 255)

    frame[y1:y2, x1:x2] = roi.astype(np.uint8)

    # ── Eye-region extra boost ──────────────────────────────────────
    # kps = [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], ...]
    if kps is not None and len(kps) >= 2:
        fw, fh = x2 - x1, y2 - y1
        eye_r  = max(8, int(min(fw, fh) * 0.16))   # radius ~ 16% of face
        for ex, ey in kps[:2]:                       # left + right eye
            ex, ey = int(ex), int(ey)
            ex1 = max(0, ex - eye_r);  ey1 = max(0, ey - eye_r)
            ex2 = min(frame.shape[1], ex + eye_r)
            ey2 = min(frame.shape[0], ey + eye_r)
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            eroi = frame[ey1:ey2, ex1:ex2].astype(np.float32)
            eblr = cv2.GaussianBlur(eroi, (0, 0), 0.3)
            eroi = np.clip(eroi + 1.2 * (eroi - eblr), 0, 255)
            frame[ey1:ey2, ex1:ex2] = eroi.astype(np.uint8)

    return frame




def _face_swap_insightface(video_path, face_image_path, output_name):
    """
    GAN face swap — InsightFace + inswapper_128.

    • EMA bbox smoothing     →  eliminates flicker
    • Reinhard colour match  →  skin-tone alignment
    • Feathered alpha blend  →  zero-seam edges
    • Multi-pass sharpening  →  counters 128×128 model softness
    """
    import insightface, time as _time
    from insightface.app import FaceAnalysis

    CPU = ['CPUExecutionProvider']

    DETECT_EVERY  = 4     # run face detection every N frames

    # ── 1. Face analyser ─────────────────────────────────────────────
    model_name = 'buffalo_l' if os.path.isdir(
        os.path.expanduser('~/.insightface/models/buffalo_l')) else 'buffalo_sc'
    app = FaceAnalysis(name=model_name, providers=CPU)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # ── 2. Source face embedding ──────────────────────────────────────
    src_img = cv2.imread(face_image_path)
    if src_img is None:
        src_img = cv2.cvtColor(np.array(Image.open(face_image_path).convert('RGB')),
                               cv2.COLOR_RGB2BGR)
    src_faces = app.get(src_img)
    if not src_faces:
        src_faces = app.get(cv2.resize(src_img, (640, 640)))
    if not src_faces:
        return "❌ No face detected in source image. Use a clear front-facing photo."
    src_face     = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    src_embedding = src_face.normed_embedding   # used for hero detection

    # ── 3. Swap model ────────────────────────────────────────────────
    model_path = _ensure_swap_model()
    swapper    = insightface.model_zoo.get_model(model_path, providers=CPU)
    print(f"✅ Model: {os.path.basename(model_path)}")

    # ── 4. Open video ────────────────────────────────────────────────
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    secs  = total / fps if fps else 0

    tmp    = out_path("_tmp_swap.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    print(f"🔄 Swapping {total} frames @ {fps:.0f}fps …")

    EMA          = 0.25
    last_face    = None
    ema_bbox     = None
    cached_mask  = None
    t0           = _time.time()

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Face detection ────────────────────────────────────────
        if i % DETECT_EVERY == 0:
            faces = app.get(frame)
            if faces:
                if len(faces) == 1:
                    raw = faces[0]
                else:
                    raw = max(faces, key=lambda f: (
                        float(np.dot(f.normed_embedding, src_embedding))
                        if hasattr(f, 'normed_embedding') and f.normed_embedding is not None
                        else (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
                    ))

                if ema_bbox is None:
                    ema_bbox    = raw.bbox.copy().astype(np.float32)
                    last_face   = raw
                    cached_mask = None
                else:
                    prev_ema = ema_bbox.copy()
                    ema_bbox = EMA * raw.bbox.astype(np.float32) + (1 - EMA) * ema_bbox
                    if np.max(np.abs(ema_bbox - prev_ema)) > 6:
                        cached_mask = None
                    last_face      = raw
                    last_face.bbox = ema_bbox.astype(np.float32)

        if last_face is not None:
            b   = ema_bbox.astype(int) if ema_bbox is not None else last_face.bbox.astype(int)
            bx1 = max(0, b[0]);  by1 = max(0, b[1])
            bx2 = min(W, b[2]);  by2 = min(H, b[3])
            fw  = bx2 - bx1;     fh  = by2 - by1

            try:
                swapped = swapper.get(frame, last_face, src_face, paste_back=True)

                if fw > 8 and fh > 8:
                    # Skin-tone colour correction
                    orig_roi  = frame  [by1:by2, bx1:bx2]
                    swap_roi  = swapped[by1:by2, bx1:bx2]
                    corrected = _reinhard_color_match(swap_roi, orig_roi)
                    swapped[by1:by2, bx1:bx2] = cv2.addWeighted(
                        corrected, 0.40, swap_roi, 0.60, 0)

                    # Feathered alpha blend
                    if cached_mask is None:
                        mask_roi = np.zeros((fh, fw), dtype=np.uint8)
                        cx, cy   = fw // 2, fh // 2
                        cv2.ellipse(mask_roi, (cx, cy),
                                    (max(4, int(cx*0.88)), max(4, int(cy*0.88))),
                                    0, 0, 360, 255, -1)
                        cached_mask = mask_roi

                    smooth_mask = _build_face_mask(last_face, W, H, feather=71)
                    m3    = smooth_mask[:, :, np.newaxis]
                    frame = (swapped.astype(np.float32) * m3 +
                             frame.astype(np.float32)   * (1 - m3))
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                    frame = _sharpen_face(frame, bx1, by1, bx2, by2,
                                          strength=0.65,
                                          kps=last_face.kps if hasattr(last_face,'kps') else None)

            except Exception as exc:
                print(f"   ⚠️  swap error frame {i}: {exc}")

        writer.write(frame)
        i += 1

        if i % 30 == 0:
            elapsed = _time.time() - t0
            spf     = elapsed / i
            eta     = int(spf * max(0, total - i))
            print(f"   frame {i}/{total}  {spf:.2f}s/f  ETA ~{eta//60}m{eta%60:02d}s")

    cap.release()
    writer.release()
    print(f"✅ Swap done in {_time.time() - t0:.0f}s")

    # ── 6. Detect audio once (needed for both mux paths) ────────────
    out = out_path(output_name)
    has_audio = bool(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True).stdout.strip())

    # ── 7. Optional: CodeFormer / GFPGAN face enhancement pass ──────
    print('\n── Face Enhancement Pass ──────────────────────────────')
    enhance_out = out_path('_tmp_enhanced_faces.mp4')
    restorer    = _load_face_restorer()
    if restorer is not None:
        etype = type(restorer).__name__
        engine_name = ('CodeFormer' if 'CF' in etype or 'Code' in etype
                       else 'GFPGAN')
        print(f'✨ Running {engine_name} ({etype}) face enhancement pass …')
        cap2   = cv2.VideoCapture(tmp)
        fps2   = cap2.get(cv2.CAP_PROP_FPS) or fps
        W2     = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        H2     = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        wrt2   = cv2.VideoWriter(enhance_out,
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps2, (W2, H2))
        gi = 0
        while True:
            ret2, frm2 = cap2.read()
            if not ret2:
                break
            # pass face_bbox so OpenCV fallback knows where to enhance
            bbox = tuple(ema_bbox.astype(int)) if ema_bbox is not None else None
            frm2 = _gfpgan_enhance_frame(restorer, frm2, face_bbox=bbox)
            wrt2.write(frm2)
            gi += 1
            if gi % 30 == 0:
                print(f'   {engine_name} {gi}/{total2} frames …')
        cap2.release()
        wrt2.release()
        # point tmp at the enhanced output for the final ffmpeg mux
        try:
            os.remove(tmp)
        except Exception:
            pass
        tmp = enhance_out
    else:
        print('ℹ️  CodeFormer model not downloaded yet — skipping enhancement pass.')
        print('   → Run option 20 "Download CodeFormer ONNX" from the agent menu once.')
        print('   It takes ~2 min to download 180 MB, then every swap is sharp.')

    # ── 8. Mux audio back ────────────────────────────────────────────
    cmd = ['ffmpeg', '-y', '-i', tmp]
    if has_audio:
        cmd += ['-i', video_path,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
                '-c:a', 'aac',
                out]
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '17', out]

    subprocess.run(cmd, check=True, capture_output=True)
    try:
        os.remove(tmp)
    except Exception:
        pass

    return done(out)


# ─────────────────────────────────────────────────────────────────────────────
# ④-C  FACE ENHANCEMENT  (CodeFormer / GFPGAN post-processing)
#       Runs AFTER face swap to sharpen, restore skin texture, fix artifacts.
#       CodeFormer is tried first (newer, better, no basicsr conflicts).
#       GFPGAN is the fallback. OpenCV CLAHE is always the last resort.
# ─────────────────────────────────────────────────────────────────────────────

# ── CodeFormer paths ─────────────────────────────────────────────────────────
_CODEFORMER_MODEL_DIR  = os.path.expanduser('~/.cache/codeformer')
_CODEFORMER_MODEL_PATH = os.path.join(_CODEFORMER_MODEL_DIR, 'codeformer.onnx')
# Try multiple mirrors in order — different facefusion release tags
_CODEFORMER_URLS = [
    'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx',
    'https://github.com/facefusion/facefusion-assets/releases/download/models-2.3.0/codeformer.onnx',
    'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
    'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/codeformer.onnx',
]
_CODEFORMER_URL = _CODEFORMER_URLS[0]   # kept for backward compat
_codeformer_restorer   = None   # singleton

# ── GFPGAN paths (fallback) ──────────────────────────────────────────────────
_GFPGAN_MODEL_DIR  = os.path.expanduser('~/.cache/gfpgan')
_GFPGAN_MODEL_PATH = os.path.join(_GFPGAN_MODEL_DIR, 'GFPGANv1.4.pth')
_GFPGAN_URL        = ('https://github.com/TencentARC/GFPGAN/releases/'
                      'download/v1.3.4/GFPGANv1.4.pth')
_gfpgan_restorer   = None   # module-level singleton — load once, reuse


# ── Completescandir stub content ─────────────────────────────────────────────
_BASICSR_UTILS_CONTENT = '''
import os as _os
import cv2 as _cv2
from .registry import ARCH_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY

def imwrite(img, path, **kw): return _cv2.imwrite(path, img)

class ProgressBar:
    def __init__(self, task_num=0): pass
    def update(self, msg=""): pass

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find files. Minimal stub for gfpgan compatibility."""
    root = dir_path
    def _scan(dp):
        for entry in _os.scandir(dp):
            if entry.is_file() and not entry.name.startswith("."):
                ret = entry.path if full_path else _os.path.relpath(entry.path, root)
                if suffix is None or ret.endswith(suffix):
                    yield ret
            elif entry.is_dir() and recursive:
                yield from _scan(entry.path)
    return _scan(dir_path)
'''


def patch_basicsr_stub():
    """
    Patch the installed basicsr stub:
      - Rewrites basicsr/utils/__init__.py with scandir + registry exports
      - Creates basicsr/ops/fused_act.py and basicsr/ops/upfirdn2d.py
        (pure-Python stubs so gfpgan/archs/stylegan2_bilinear_arch.py can
        be imported without the compiled C extension)
    Call this after install_face_enhancer() if GFPGAN still fails.
    Returns a status string.
    """
    global _gfpgan_restorer
    results = []
    try:
        import site as _site
        sp  = _site.getsitepackages()[0]
        bsr = os.path.join(sp, 'basicsr')
        if not os.path.isdir(bsr):
            return '⚠️  basicsr not found in site-packages — run Install Face Enhancer first'

        # ── 1. utils/__init__.py (scandir + registry) ──────────────────
        dst = os.path.join(bsr, 'utils', '__init__.py')
        with open(dst, 'w') as f:
            f.write(_BASICSR_UTILS_CONTENT)
        results.append('✅  basicsr/utils patched (scandir added)')

        # ── 2. ops/ — pure Python stubs for fused_act + upfirdn2d ──────
        ops_dir = os.path.join(bsr, 'ops')
        os.makedirs(ops_dir, exist_ok=True)

        with open(os.path.join(ops_dir, '__init__.py'), 'w') as f:
            f.write('# basicsr ops stub — pure Python fallback\n')

        with open(os.path.join(ops_dir, 'fused_act.py'), 'w') as f:
            f.write(
                '# Pure-Python stub: GFPGANv1Clean never calls StyleGAN2 layers\n'
                'import torch, torch.nn as nn, torch.nn.functional as F\n\n'
                'def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):\n'
                '    if bias is not None:\n'
                '        rest = [1] * (input.ndim - 1 - bias.ndim)\n'
                '        return F.leaky_relu(input + bias.view(1,-1,*rest),\n'
                '                            negative_slope=negative_slope) * scale\n'
                '    return F.leaky_relu(input, negative_slope=negative_slope) * scale\n\n'
                'class FusedLeakyReLU(nn.Module):\n'
                '    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):\n'
                '        super().__init__()\n'
                '        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None\n'
                '        self.negative_slope = negative_slope; self.scale = scale\n'
                '    def forward(self, x):\n'
                '        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)\n'
            )

        with open(os.path.join(ops_dir, 'upfirdn2d.py'), 'w') as f:
            f.write(
                '# Pure-Python stub: never called by GFPGANv1Clean\n'
                'import torch, torch.nn.functional as F\n\n'
                'def upfirdn2d(input, kernel, up=1, down=1, pad=(0,0)):\n'
                '    out = input\n'
                '    if up > 1:\n'
                '        B,C,H,W = out.shape\n'
                '        out = out.view(B,C,H,1,W,1)\n'
                '        out = F.pad(out,[0,up-1,0,0,0,up-1])\n'
                '        out = out.view(B,C,H*up,W*up)\n'
                '    if pad[0]+pad[1]>0: out=F.pad(out,[pad[0],pad[1],pad[0],pad[1]])\n'
                '    if kernel is not None:\n'
                '        k=kernel.float(); k=k/k.sum()\n'
                '        k2d=k.unsqueeze(0).unsqueeze(0).expand(out.shape[1],1,*k.shape[-2:])\n'
                '        out=F.conv2d(out,k2d,padding=0,groups=out.shape[1])\n'
                '    if down>1: out=out[:,:,::down,::down]\n'
                '    return out\n'
            )

        results.append('✅  basicsr/ops patched (fused_act + upfirdn2d stubs written)')

        # ── 3. archs/arch_util.py — default_init_weights ───────────────
        # stylegan2_clean_arch.py does `from basicsr.archs.arch_util import
        # default_init_weights` at module load time.
        with open(os.path.join(bsr, 'archs', 'arch_util.py'), 'w') as f:
            f.write(
                '# Pure-Python stub for basicsr.archs.arch_util\n'
                'import torch.nn as nn, torch.nn.init as init\n\n'
                'def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):\n'
                '    """Weight init stub — weights are loaded from checkpoint anyway."""\n'
                '    if not isinstance(module_list, list):\n'
                '        module_list = [module_list]\n'
                '    for module in module_list:\n'
                '        for m in module.modules():\n'
                '            if isinstance(m, (nn.Conv2d, nn.Linear)):\n'
                '                init.kaiming_normal_(m.weight, **kwargs)\n'
                '                m.weight.data *= scale\n'
                '                if m.bias is not None:\n'
                '                    m.bias.data.fill_(bias_fill)\n\n'
                'def make_layer(basic_block, num_basic_block, **kwarg):\n'
                '    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]\n'
                '    return nn.Sequential(*layers)\n\n'
                'class ResidualBlockNoBN(nn.Module):\n'
                '    def __init__(self, num_feat=64, res_scale=1):\n'
                '        super().__init__()\n'
                '        self.res_scale = res_scale\n'
                '        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)\n'
                '        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)\n'
                '        self.relu  = nn.ReLU(inplace=True)\n'
                '    def forward(self, x):\n'
                '        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))\n'
            )
        results.append('✅  basicsr/archs/arch_util.py stub written')

        # ── 4. archs/stylegan2_arch.py — gfpganv1_arch needs these classes ──
        # gfpgan/archs/__init__.py imports ALL arch files including gfpganv1_arch.py.
        # That file imports ConvLayer, EqualConv2d, etc. from stylegan2_arch.
        # We only USE GFPGANv1Clean, but the whole archs package must be importable.
        with open(os.path.join(bsr, 'archs', 'stylegan2_arch.py'), 'w') as f:
            f.write(
                '# Stub for basicsr.archs.stylegan2_arch — import-only, never called\n'
                'import torch.nn as nn, torch.nn.functional as F\n\n'
                'class ScaledLeakyReLU(nn.Module):\n'
                '    def __init__(self, negative_slope=0.2):\n'
                '        super().__init__(); self.ns=negative_slope\n'
                '    def forward(self,x): return F.leaky_relu(x,self.ns)*(2**0.5)\n\n'
                'class EqualLinear(nn.Module):\n'
                '    def __init__(self,in_f,out_f,bias=True,bias_init=0,lr_mul=1,activation=None):\n'
                '        super().__init__()\n'
                '        self.linear=nn.Linear(in_f,out_f,bias=bias)\n'
                '        if bias: self.linear.bias.data.fill_(bias_init)\n'
                '    def forward(self,x): return self.linear(x)\n\n'
                'class EqualConv2d(nn.Module):\n'
                '    def __init__(self,in_c,out_c,k,stride=1,padding=0,bias=True):\n'
                '        super().__init__()\n'
                '        self.conv=nn.Conv2d(in_c,out_c,k,stride,padding,bias=bias)\n'
                '    def forward(self,x): return self.conv(x)\n\n'
                'class ConvLayer(nn.Sequential):\n'
                '    def __init__(self,in_c,out_c,k,downsample=False,bias=True,activate=True):\n'
                '        layers=[]\n'
                '        s=2 if downsample else 1\n'
                '        layers.append(EqualConv2d(in_c,out_c,k,stride=s,padding=k//2,bias=bias and not activate))\n'
                '        if activate: layers.append(ScaledLeakyReLU())\n'
                '        super().__init__(*layers)\n\n'
                'class ResBlock(nn.Module):\n'
                '    def __init__(self,in_c,out_c,mode="down"):\n'
                '        super().__init__()\n'
                '        self.c1=ConvLayer(in_c,in_c,3)\n'
                '        self.c2=ConvLayer(in_c,out_c,3,downsample=(mode=="down"))\n'
                '        self.skip=ConvLayer(in_c,out_c,1,downsample=(mode=="down"),activate=False,bias=False)\n'
                '    def forward(self,x): return (self.c2(self.c1(x))+self.skip(x))/(2**0.5)\n\n'
                'class StyleGAN2Generator(nn.Module):\n'
                '    def __init__(self,*a,**kw): super().__init__()\n'
                '    def forward(self,*a,**kw): return None\n'
            )
        results.append('✅  basicsr/archs/stylegan2_arch.py stub written')

        # Force GFPGAN to reload on next face swap
        _gfpgan_restorer = None
        return '\n'.join(results) + '\n✅  All stubs ready — GFPGAN should load now!'
    except Exception as e:
        return f'❌  patch failed: {e}'


def install_face_enhancer():
    """
    Install GFPGAN on macOS (handles Apple Silicon + system Python quirks).
    Strategy: install PyTorch first (required by basicsr/gfpgan), then gfpgan.
    Returns a status string.
    """
    lines = []

    def pip(*args):
        cmd = ['pip3', 'install', '--break-system-packages', '-q'] + list(args)
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0, r.stderr.strip()

    # ── Step 1: PyTorch (required by basicsr/gfpgan, has pre-built ARM wheels) ──
    try:
        import torch
        lines.append(f'✅  torch already installed ({torch.__version__})')
    except ImportError:
        lines.append('📥  Installing PyTorch (Apple Silicon build) …')
        ok, err = pip('torch', 'torchvision', '--index-url',
                      'https://download.pytorch.org/whl/cpu')
        lines.append('✅  torch installed' if ok else f'⚠️  torch: {err[:120]}')

    # ── Step 2: facexlib (lightweight, usually installs fine) ───────
    ok, err = pip('facexlib')
    lines.append('✅  facexlib' if ok else f'⚠️  facexlib: {err[:80]}')

    # ── Step 3: basicsr — clone from git, patch __version__ bug, install ──
    # ── Step 3: basicsr stub ─────────────────────────────────────────
    # basicsr's pyproject.toml build fails on all macOS torch nightly builds.
    # gfpgan only uses 3 things from basicsr: ARCH_REGISTRY, imwrite,
    # load_file_from_url. We write a minimal stub directly to site-packages.
    lines.append('🔧  Writing minimal basicsr stub …')
    basicsr_ok = False
    try:
        import site as _site
        sp = _site.getsitepackages()[0]
        bsr = os.path.join(sp, 'basicsr')
        os.makedirs(os.path.join(bsr, 'utils'), exist_ok=True)
        os.makedirs(os.path.join(bsr, 'archs'), exist_ok=True)

        # basicsr/__init__.py
        with open(os.path.join(bsr, '__init__.py'), 'w') as f:
            f.write('# basicsr stub for gfpgan compatibility\n__version__ = "1.4.2"\n')

        # basicsr/utils/registry.py — ARCH_REGISTRY is all gfpgan needs
        with open(os.path.join(bsr, 'utils', 'registry.py'), 'w') as f:
            f.write('''
class _Registry:
    def __init__(self, name): self._name = name; self._map = {}
    def register(self, obj=None):
        if obj is None:
            def deco(fn): self._map[fn.__name__] = fn; return fn
            return deco
        self._map[obj.__name__] = obj; return obj
    def get(self, name): return self._map.get(name)
    def __contains__(self, name): return name in self._map
    def __getitem__(self, name): return self._map[name]

ARCH_REGISTRY   = _Registry("arch")
LOSS_REGISTRY   = _Registry("loss")
METRIC_REGISTRY = _Registry("metric")
MODEL_REGISTRY  = _Registry("model")
''')

        # basicsr/utils/__init__.py
        with open(os.path.join(bsr, 'utils', '__init__.py'), 'w') as f:
            f.write('''
import os as _os
import cv2 as _cv2
from .registry import ARCH_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY

def imwrite(img, path, **kw): return _cv2.imwrite(path, img)

class ProgressBar:
    def __init__(self, task_num=0): pass
    def update(self, msg=""): pass

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find files. Minimal stub for gfpgan compatibility."""
    root = dir_path
    def _scan(dp):
        for entry in _os.scandir(dp):
            if entry.is_file() and not entry.name.startswith("."):
                ret = entry.path if full_path else _os.path.relpath(entry.path, root)
                if suffix is None or ret.endswith(suffix):
                    yield ret
            elif entry.is_dir() and recursive:
                yield from _scan(entry.path)
    return _scan(dir_path)
''')

        # basicsr/utils/download_util.py
        with open(os.path.join(bsr, 'utils', 'download_util.py'), 'w') as f:
            f.write('''
import os, subprocess

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "basicsr")
    os.makedirs(model_dir, exist_ok=True)
    fname = file_name or os.path.basename(url.split("?")[0])
    path  = os.path.join(model_dir, fname)
    if not os.path.exists(path):
        subprocess.run(["curl", "-L", "-o", path, url])
    return path
''')

        # basicsr/archs/__init__.py
        with open(os.path.join(bsr, 'archs', '__init__.py'), 'w') as f:
            f.write('from basicsr.utils.registry import ARCH_REGISTRY\n')

        # basicsr/ops/ — stub for fused_act and upfirdn2d
        # These are normally compiled C/CUDA extensions that fail to build on
        # Python 3.13 / Apple Silicon. GFPGANv1Clean never calls them at
        # runtime, but gfpgan/archs/__init__.py imports stylegan2_bilinear_arch
        # which triggers these imports at module load time.
        os.makedirs(os.path.join(bsr, 'ops'), exist_ok=True)
        with open(os.path.join(bsr, 'ops', '__init__.py'), 'w') as f:
            f.write('# basicsr ops stub — pure Python fallbacks, no C extension needed\n')

        with open(os.path.join(bsr, 'ops', 'fused_act.py'), 'w') as f:
            f.write('''
# Pure-Python stub for basicsr.ops.fused_act
# GFPGANv1Clean does NOT use StyleGAN2 layers, so this is import-only.
import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - 1 - bias.ndim)
        return F.leaky_relu(
            input + bias.view(1, -1, *rest_dim),
            negative_slope=negative_slope
        ) * scale
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias,
                                self.negative_slope, self.scale)
''')

        with open(os.path.join(bsr, 'ops', 'upfirdn2d.py'), 'w') as f:
            f.write('''
# Pure-Python stub for basicsr.ops.upfirdn2d
# GFPGANv1Clean does NOT use StyleGAN2 layers, so this is import-only.
import torch
import torch.nn.functional as F

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Simplified CPU fallback. Only called by StyleGAN2 which GFPGAN Clean avoids."""
    out = input
    if up > 1:
        B, C, H, W = out.shape
        out = out.view(B, C, H, 1, W, 1)
        out = F.pad(out, [0, up - 1, 0, 0, 0, up - 1])
        out = out.view(B, C, H * up, W * up)
    p = pad[0] + pad[1]
    if p > 0:
        out = F.pad(out, [pad[0], pad[1], pad[0], pad[1]])
    if kernel is not None:
        k = kernel.float()
        k = k / k.sum()
        k2d = k.unsqueeze(0).unsqueeze(0)
        k2d = k2d.expand(out.shape[1], 1, *k2d.shape[2:])
        out = F.conv2d(out, k2d, padding=0, groups=out.shape[1])
    if down > 1:
        out = out[:, :, ::down, ::down]
    return out
''')

        # basicsr/archs/arch_util.py — stylegan2_clean_arch needs default_init_weights
        with open(os.path.join(bsr, 'archs', 'arch_util.py'), 'w') as f:
            f.write(
                '# Pure-Python stub for basicsr.archs.arch_util\n'
                'import torch.nn as nn, torch.nn.init as init\n\n'
                'def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):\n'
                '    if not isinstance(module_list, list): module_list=[module_list]\n'
                '    for module in module_list:\n'
                '        for m in module.modules():\n'
                '            if isinstance(m,(nn.Conv2d,nn.Linear)):\n'
                '                init.kaiming_normal_(m.weight,**kwargs)\n'
                '                m.weight.data*=scale\n'
                '                if m.bias is not None: m.bias.data.fill_(bias_fill)\n\n'
                'def make_layer(basic_block,num_basic_block,**kwarg):\n'
                '    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])\n\n'
                'class ResidualBlockNoBN(nn.Module):\n'
                '    def __init__(self,num_feat=64,res_scale=1):\n'
                '        super().__init__()\n'
                '        self.res_scale=res_scale\n'
                '        self.conv1=nn.Conv2d(num_feat,num_feat,3,1,1)\n'
                '        self.conv2=nn.Conv2d(num_feat,num_feat,3,1,1)\n'
                '        self.relu=nn.ReLU(inplace=True)\n'
                '    def forward(self,x): return x+self.res_scale*self.conv2(self.relu(self.conv1(x)))\n'
            )

        # basicsr/archs/stylegan2_arch.py — gfpganv1_arch.py imports from here
        with open(os.path.join(bsr, 'archs', 'stylegan2_arch.py'), 'w') as f:
            f.write(
                '# Stub — import-only, GFPGANv1Clean never instantiates these\n'
                'import torch.nn as nn, torch.nn.functional as F\n\n'
                'class ScaledLeakyReLU(nn.Module):\n'
                '    def __init__(self,ns=0.2): super().__init__(); self.ns=ns\n'
                '    def forward(self,x): return F.leaky_relu(x,self.ns)*(2**0.5)\n\n'
                'class EqualLinear(nn.Module):\n'
                '    def __init__(self,in_f,out_f,bias=True,bias_init=0,lr_mul=1,activation=None):\n'
                '        super().__init__()\n'
                '        self.linear=nn.Linear(in_f,out_f,bias=bias)\n'
                '        if bias: self.linear.bias.data.fill_(bias_init)\n'
                '    def forward(self,x): return self.linear(x)\n\n'
                'class EqualConv2d(nn.Module):\n'
                '    def __init__(self,in_c,out_c,k,stride=1,padding=0,bias=True):\n'
                '        super().__init__()\n'
                '        self.conv=nn.Conv2d(in_c,out_c,k,stride,padding,bias=bias)\n'
                '    def forward(self,x): return self.conv(x)\n\n'
                'class ConvLayer(nn.Sequential):\n'
                '    def __init__(self,in_c,out_c,k,downsample=False,bias=True,activate=True):\n'
                '        s=2 if downsample else 1\n'
                '        layers=[EqualConv2d(in_c,out_c,k,stride=s,padding=k//2,bias=bias and not activate)]\n'
                '        if activate: layers.append(ScaledLeakyReLU())\n'
                '        super().__init__(*layers)\n\n'
                'class ResBlock(nn.Module):\n'
                '    def __init__(self,in_c,out_c,mode="down"):\n'
                '        super().__init__()\n'
                '        self.c1=ConvLayer(in_c,in_c,3)\n'
                '        self.c2=ConvLayer(in_c,out_c,3,downsample=(mode=="down"))\n'
                '        self.skip=ConvLayer(in_c,out_c,1,downsample=(mode=="down"),activate=False,bias=False)\n'
                '    def forward(self,x): return (self.c2(self.c1(x))+self.skip(x))/(2**0.5)\n\n'
                'class StyleGAN2Generator(nn.Module):\n'
                '    def __init__(self,*a,**kw): super().__init__()\n'
                '    def forward(self,*a,**kw): return None\n'
            )

        lines.append('✅  basicsr stub written (ops + archs/arch_util + archs/stylegan2_arch)')
        basicsr_ok = True
    except Exception as e:
        lines.append(f'⚠️  basicsr stub failed: {e}')

    # ── Step 4: gfpgan (now basicsr is present) ──────────────────────
    ok, err = pip('gfpgan', '--no-build-isolation')
    if not ok:
        ok, err = pip('gfpgan==1.3.8', '--no-deps')
    lines.append('✅  gfpgan' if ok else f'⚠️  gfpgan: {err[:120]}')

    # ── Step 5: model weights (already downloaded = skip) ───────────
    os.makedirs(_GFPGAN_MODEL_DIR, exist_ok=True)
    if os.path.exists(_GFPGAN_MODEL_PATH) and \
            os.path.getsize(_GFPGAN_MODEL_PATH) > 200_000_000:
        mb = os.path.getsize(_GFPGAN_MODEL_PATH) // 1_000_000
        lines.append(f'✅  GFPGANv1.4.pth ready ({mb} MB)')
    else:
        lines.append('📥  Downloading GFPGANv1.4 (~350 MB) …')
        r = subprocess.run(
            ['curl', '-L', '--progress-bar', '-o', _GFPGAN_MODEL_PATH, _GFPGAN_URL])
        if r.returncode == 0 and os.path.getsize(_GFPGAN_MODEL_PATH) > 200_000_000:
            lines.append('✅  GFPGANv1.4.pth downloaded')
        else:
            lines.append('❌  Download failed — check internet')

    # ── Auto-patch the installed basicsr stub (adds scandir etc.) ────
    lines.append('\n🔧  Auto-patching basicsr stub …')
    lines.append(patch_basicsr_stub())

    # ── Final check: actually try to load GFPGAN ─────────────────────
    global _gfpgan_restorer
    _gfpgan_restorer = None   # force a fresh load attempt
    lines.append('🔍  Testing GFPGAN load …')
    restorer = _load_gfpgan()
    if restorer is not None:
        ename = type(restorer).__name__
        lines.append(f'✅  GFPGAN loaded successfully! Engine: {ename}')
        lines.append('   Sharp eyes + restored detail on every face swap.')
    else:
        lines.append('⚠️  GFPGAN did not load — will use OpenCV multi-pass fallback.')
        lines.append('   (OpenCV fallback still applies CLAHE + bilateral + 2× unsharp mask)')
    lines.append(f'   Model: {_GFPGAN_MODEL_PATH}')
    try:
        lines.append(f'   Size:  {os.path.getsize(_GFPGAN_MODEL_PATH)//1_000_000} MB')
    except Exception:
        pass

    return '\n'.join(lines)


def _cf_get_face_landmarks(app, bgr_img):
    """
    Use InsightFace (already installed for face swap) to get 5-point landmarks.
    Returns list of (kps_5x2, bbox) or empty list if no face found.
    """
    faces = app.get(bgr_img)
    results = []
    for face in faces:
        if hasattr(face, 'kps') and face.kps is not None:
            results.append((face.kps.astype(np.float32), face.bbox))
    return results


# Standard 5-point 512×512 face template used by GFPGAN / CodeFormer
_CF_FACE_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19366],
    [256.63416, 314.01935],
    [201.26117, 371.41046],
    [313.08905, 371.15118],
], dtype=np.float32)


def _cf_align_face(bgr_img, kps, face_size=512):
    """
    Affine-warp a face to face_size×face_size using 5-point landmarks.
    Returns (warped_bgr, inverse_matrix).
    """
    template = _CF_FACE_TEMPLATE_512 * (face_size / 512.0)
    mat, _ = cv2.estimateAffinePartial2D(kps, template,
                                          method=cv2.LMEDS)
    warped = cv2.warpAffine(bgr_img, mat, (face_size, face_size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    # inverse matrix for pasting back
    inv_mat = cv2.invertAffineTransform(mat)
    return warped, inv_mat


def _cf_paste_face_back(bgr_frame, restored_face, inv_mat):
    """
    Paste restored 512×512 face back into the full frame using inverse affine.
    Uses seamless alpha blending at the boundary so edges don't look pasted.
    """
    H, W = bgr_frame.shape[:2]
    # Warp restored face back to original position
    restored_back = cv2.warpAffine(restored_face, inv_mat, (W, H),
                                    flags=cv2.INTER_LINEAR)
    # Build a soft mask for the face region only
    mask = np.ones((512, 512), dtype=np.float32)
    # erode + gaussian so edges blend rather than hard-cut
    mask = cv2.erode(mask, np.ones((32, 32), np.uint8))
    mask = cv2.GaussianBlur(mask, (65, 65), 20)
    mask_back = cv2.warpAffine(mask, inv_mat, (W, H))
    mask_3ch  = mask_back[:, :, np.newaxis]
    blended   = (restored_back.astype(np.float32) * mask_3ch
                 + bgr_frame.astype(np.float32) * (1 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _load_codeformer():
    """
    Load CodeFormer ONNX face restorer.
    Uses onnxruntime (already installed with insightface) + insightface for alignment.
    ZERO new pip installs required.
    Downloads codeformer.onnx (~180 MB) on first use.
    Returns a restorer with .enhance(bgr_img) → (None, None, enhanced_bgr).
    """
    global _codeformer_restorer
    if _codeformer_restorer is not None:
        return _codeformer_restorer

    try:
        import onnxruntime as _ort
        print('  ✔  onnxruntime available')
    except ImportError:
        print('ℹ️  onnxruntime not found — cannot use CodeFormer ONNX')
        return None

    try:
        # ── 1. Check ONNX model exists (don't auto-download during face swap) ─
        os.makedirs(_CODEFORMER_MODEL_DIR, exist_ok=True)
        onnx_path = os.path.join(_CODEFORMER_MODEL_DIR, 'codeformer.onnx')

        if not os.path.exists(onnx_path):
            print('ℹ️  CodeFormer model not downloaded yet — run option 20 first.')
            return None

        # Validate: must be >10 MB and load as valid protobuf
        sz = os.path.getsize(onnx_path)
        if sz < 10_000_000:
            print(f'⚠️  codeformer.onnx too small ({sz//1024} KB) — re-run option 20.')
            return None
        # Quick protobuf sanity-check using onnx if available, else just byte check
        try:
            import onnx as _onnx
            _onnx.checker.check_model(onnx_path)
        except ImportError:
            with open(onnx_path, 'rb') as _f:
                _hdr = _f.read(16)
            if _hdr.startswith(b'<') or _hdr.startswith(b'<!'):
                print('⚠️  codeformer.onnx is an HTML page — deleting, re-run option 20.')
                os.remove(onnx_path)
                return None
        except Exception as _ve:
            print(f'⚠️  codeformer.onnx is corrupt ({_ve}) — deleting, re-run option 20.')
            os.remove(onnx_path)
            return None

        # ── 2. Create ONNX session (CoreML on Apple Silicon, CPU fallback) ─
        providers = []
        available = _ort.get_available_providers()
        if 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
            print('  🍎  Using CoreML (Apple Silicon MPS acceleration)')
        providers.append('CPUExecutionProvider')

        sess_opts = _ort.SessionOptions()
        sess_opts.log_severity_level = 3   # silence warnings
        try:
            session = _ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                             providers=providers)
        except Exception as _ort_err:
            print(f'⚠️  codeformer.onnx failed to load ({_ort_err.__class__.__name__}) — '
                  f'deleting corrupt file. Re-run option 20 to re-download.')
            try:
                os.remove(onnx_path)
            except Exception:
                pass
            return None
        input_name  = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape   # [1, 3, 512, 512]
        face_size   = int(input_shape[2]) if len(input_shape) >= 3 else 512
        print(f'  ✔  CodeFormer ONNX session ready  (input={input_name} {input_shape})')

        # ── 3. InsightFace app for landmark detection ─────────────────────
        from insightface.app import FaceAnalysis as _FA
        _cf_app = _FA(name='buffalo_l', providers=['CPUExecutionProvider'])
        _cf_app.prepare(ctx_id=0, det_size=(640, 640))
        print('  ✔  InsightFace detector ready for CodeFormer alignment')

        # ── 4. Build restorer object ──────────────────────────────────────
        class _CFOnnxRestorer:
            """
            Drop-in replace for GFPGANer.
            .enhance(bgr_img) → (None, None, enhanced_bgr)
            Uses CodeFormer ONNX + InsightFace 5-pt alignment.
            """
            def enhance(self_, bgr_img, **kw):
                faces_info = _cf_get_face_landmarks(_cf_app, bgr_img)
                if not faces_info:
                    return None, None, bgr_img   # no face detected

                result = bgr_img.copy()
                for kps, _bbox in faces_info:
                    # align → 512×512 BGR
                    warped, inv_mat = _cf_align_face(bgr_img, kps, face_size)

                    # BGR uint8 → RGB float32 NCHW, normalised [-1, 1]
                    rgb = warped[:, :, ::-1].astype(np.float32) / 255.0
                    t   = (rgb.transpose(2, 0, 1)[np.newaxis] - 0.5) / 0.5

                    # run CodeFormer ONNX
                    out = session.run([output_name], {input_name: t})[0]

                    # RGB float32 NCHW → BGR uint8
                    out_np = (out[0].transpose(1, 2, 0) * 0.5 + 0.5)
                    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
                    out_bgr = out_np[:, :, ::-1].copy()

                    # paste back with soft blending
                    result = _cf_paste_face_back(result, out_bgr, inv_mat)

                return None, None, result

        _codeformer_restorer = _CFOnnxRestorer()
        print('✅  CodeFormer ONNX loaded — Kling-level face restoration ready!')
        return _codeformer_restorer

    except Exception as e:
        import traceback
        print(f'⚠️  CodeFormer ONNX load failed: {type(e).__name__}: {e}')
        traceback.print_exc()
        return None


def _load_face_restorer():
    """
    Unified loader: tries CodeFormer ONNX first, then GFPGAN, returns None if both fail.
    CodeFormer ONNX needs only onnxruntime (already installed with insightface).
    None return means OpenCV CLAHE fallback runs automatically.
    """
    r = _load_codeformer()
    if r is not None:
        return r
    r = _load_gfpgan()
    return r


def install_codeformer():
    """
    Download the CodeFormer ONNX model AND patch GFPGAN's basicsr.ops stubs.
    Two enhancement engines after this: CodeFormer ONNX (primary) + GFPGAN (fallback).
    No new pip installs needed for CodeFormer — uses onnxruntime from insightface.
    Returns a status string suitable for display in the agent UI.
    """
    lines = ['🚀  Setting up CodeFormer + fixing GFPGAN …\n']

    # ── A. Patch basicsr ops stubs (fixes the basicsr.ops.fused_act error) ───
    lines.append('🔧  Patching basicsr ops stubs …')
    patch_result = patch_basicsr_stub()
    lines.append(patch_result)
    lines.append('')

    # ── B. CodeFormer ONNX ────────────────────────────────────────────────────
    lines.append('── CodeFormer ONNX ──────────────────────────────────')
    try:
        import onnxruntime as _ort
        lines.append(f'✅  onnxruntime {_ort.__version__} present')
    except ImportError:
        lines.append('❌  onnxruntime not found.')
        lines.append('   It should have been installed with insightface.')
        lines.append('   Try:  pip install onnxruntime --break-system-packages')
        lines.append('   (GFPGAN patch above may still work — check terminal)')
        return '\n'.join(lines)

    os.makedirs(_CODEFORMER_MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(_CODEFORMER_MODEL_DIR, 'codeformer.onnx')

    def _onnx_valid(path):
        """Return True only if the file is a loadable ONNX (full InferenceSession probe)."""
        if not os.path.exists(path) or os.path.getsize(path) < 10_000_000:
            return False
        # Quick header check first (fast-fail HTML/garbage)
        with open(path, 'rb') as f:
            hdr = f.read(4)
        if hdr.startswith(b'<') or hdr.startswith(b'{'):
            return False
        # Full protobuf load test — catches InvalidProtobuf at runtime
        try:
            _opts = _ort.SessionOptions()
            _opts.log_severity_level = 3
            _ort.InferenceSession(path, sess_options=_opts,
                                  providers=['CPUExecutionProvider'])
            return True
        except Exception:
            return False

    # Always delete a file that fails the validity check
    if os.path.exists(onnx_path) and not _onnx_valid(onnx_path):
        sz = os.path.getsize(onnx_path)
        lines.append(f'🗑  Removing invalid codeformer.onnx ({sz//1024} KB, not loadable) …')
        os.remove(onnx_path)

    onnx_ok = False
    if _onnx_valid(onnx_path):
        mb = os.path.getsize(onnx_path) // 1_000_000
        lines.append(f'✅  codeformer.onnx valid and present ({mb} MB)')
        onnx_ok = True
    else:
        # Try each mirror URL — validate bytes after each attempt
        for url in _CODEFORMER_URLS:
            short = '/'.join(url.split('/')[-3:])
            lines.append(f'📥  Trying: {short}')
            subprocess.run(['curl', '-sL', '-o', onnx_path, url])
            if _onnx_valid(onnx_path):
                mb = os.path.getsize(onnx_path) // 1_000_000
                lines.append(f'✅  Downloaded valid ONNX ({mb} MB)')
                onnx_ok = True
                break
            else:
                sz = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
                lines.append(f'   ✗ Got {sz//1024} KB (invalid) — trying next …')
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)

        if not onnx_ok:
            lines.append('⚠️  All CodeFormer ONNX mirrors returned invalid files.')
            lines.append('   GFPGAN (patched above) will be used instead.')

    # ── C. Download GFPGAN model weights (needed even for Tier-2 torch load) ──
    lines.append('')
    lines.append('── GFPGAN Model Weights ─────────────────────────────')
    os.makedirs(_GFPGAN_MODEL_DIR, exist_ok=True)
    if os.path.exists(_GFPGAN_MODEL_PATH) and os.path.getsize(_GFPGAN_MODEL_PATH) > 200_000_000:
        mb = os.path.getsize(_GFPGAN_MODEL_PATH) // 1_000_000
        lines.append(f'✅  GFPGANv1.4.pth present ({mb} MB)')
    else:
        lines.append('📥  Downloading GFPGANv1.4.pth (~350 MB) …')
        r = subprocess.run(['curl', '-L', '--progress-bar',
                            '-o', _GFPGAN_MODEL_PATH, _GFPGAN_URL])
        if os.path.exists(_GFPGAN_MODEL_PATH) and os.path.getsize(_GFPGAN_MODEL_PATH) > 200_000_000:
            mb = os.path.getsize(_GFPGAN_MODEL_PATH) // 1_000_000
            lines.append(f'✅  Downloaded ({mb} MB)')
        else:
            lines.append('❌  GFPGAN model download failed. Check internet.')

    # ── D. Diagnostic: check torch + show real errors ─────────────────────────
    lines.append('')
    lines.append('── Diagnostics ──────────────────────────────────────')
    try:
        import torch as _t
        lines.append(f'✅  PyTorch {_t.__version__} installed')
        mps = _t.backends.mps.is_available()
        lines.append(f'   MPS (Apple GPU): {"✅ yes" if mps else "❌ no (CPU only)"}')
    except ImportError:
        lines.append('❌  PyTorch NOT installed — GFPGAN Tier-2 cannot run.')
        lines.append('   Fix: pip install torch --break-system-packages')

    try:
        import facexlib  # noqa
        lines.append('✅  facexlib installed')
    except ImportError:
        lines.append('❌  facexlib NOT installed — GFPGAN Tier-2 cannot run.')
        lines.append('   Fix: pip install facexlib --break-system-packages')

    # Check the ops stub was actually written
    try:
        import site as _s
        ops_file = os.path.join(_s.getsitepackages()[0], 'basicsr', 'ops', 'fused_act.py')
        if os.path.exists(ops_file):
            lines.append(f'✅  basicsr/ops/fused_act.py stub present')
        else:
            lines.append(f'❌  basicsr/ops/fused_act.py missing — re-run option 20')
    except Exception as ex:
        lines.append(f'⚠️  Could not check ops stub: {ex}')

    # Try the import chain verbosely
    try:
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # noqa
        lines.append('✅  gfpgan.archs.gfpganv1_clean_arch imports OK')
    except Exception as ex:
        lines.append(f'❌  gfpgan arch import failed: {type(ex).__name__}: {ex}')

    # ── E. Live load tests ────────────────────────────────────────────────────
    lines.append('')
    lines.append('── Load Tests ───────────────────────────────────────')
    global _codeformer_restorer, _gfpgan_restorer
    _codeformer_restorer = None
    _gfpgan_restorer     = None

    cf = _load_codeformer()
    lines.append('✅  CodeFormer ONNX: ready!' if cf is not None
                 else 'ℹ️  CodeFormer ONNX: not loaded')

    gf = _load_gfpgan()
    lines.append('✅  GFPGAN: ready!' if gf is not None
                 else 'ℹ️  GFPGAN: not loaded')

    if cf is not None or gf is not None:
        lines.append('')
        lines.append('🎉  Face enhancement active! Sharp eyes on next face swap.')
    else:
        lines.append('')
        lines.append('⚠️  Neither enhancer loaded — see diagnostics above.')

    return '\n'.join(lines)


def _load_gfpgan():
    """
    Load GFPGAN restorer. Three-tier strategy:
      1. Full gfpgan package (if pip install worked)
      2. Direct PyTorch load using facexlib (torch installed, basicsr not needed)
      3. Return None → OpenCV fallback kicks in automatically
    """
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer

    # ── Tier 1: standard gfpgan package ─────────────────────────────
    try:
        from gfpgan import GFPGANer
        if not os.path.exists(_GFPGAN_MODEL_PATH):
            return None   # model not downloaded — silent, CodeFormer is primary
        _gfpgan_restorer = GFPGANer(
            model_path=_GFPGAN_MODEL_PATH,
            upscale=1, arch='clean',
            channel_multiplier=2, bg_upsampler=None)
        print('✅  GFPGAN (full package) loaded')
        return _gfpgan_restorer
    except Exception as e:
        # Tier-1 failures are expected when basicsr.data / ops stubs are incomplete.
        # Tier-2 torch inference handles it — no need to show this error.
        pass

    # ── Tier 2: direct torch inference using gfpgan arch + facexlib ──
    # GFPGANv1Clean is self-contained inside the gfpgan package.
    # We load the .pth weights with torch directly, bypassing GFPGANer entirely.
    # FaceRestoreHelper (facexlib) handles face alignment to 512x512.
    try:
        import torch as _torch
        import importlib, inspect
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        print('  ✔  GFPGANv1Clean import OK')

        if not os.path.exists(_GFPGAN_MODEL_PATH):
            print(f'  ✘  model file not found: {_GFPGAN_MODEL_PATH}')
        else:
            _device = _torch.device('mps' if _torch.backends.mps.is_available()
                                    else 'cpu')
            print(f'  🔄  Loading GFPGANv1.4 weights on {_device} …')

            net_g = GFPGANv1Clean(
                out_size=512, num_style_feat=512, channel_multiplier=2,
                decoder_load_path=None, fix_decoder=False,
                num_mlp=8, input_is_latent=True, different_w=True,
                narrow=1, sft_half=True)
            weights = _torch.load(_GFPGAN_MODEL_PATH,
                                  map_location=_device, weights_only=False)
            state  = weights.get('params_ema', weights)
            missing, unexpected = net_g.load_state_dict(state, strict=False)
            net_g.eval().to(_device)
            print(f'  ✔  net_g loaded  (missing={len(missing)}, unexpected={len(unexpected)})')

            # ── FaceRestoreHelper — robust across facexlib versions ──────
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            print('  ✔  FaceRestoreHelper import OK')

            # Probe which signature get_face_landmarks_5 accepts
            _glm5_sig = inspect.signature(
                FaceRestoreHelper.get_face_landmarks_5).parameters
            _has_resize       = 'resize'            in _glm5_sig
            _has_eye_dist     = 'eye_dist_threshold' in _glm5_sig

            face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model='retinaface_resnet50', save_ext='png',
                use_parse=True, device=_device)
            print(f'  ✔  FaceRestoreHelper ready  '
                  f'(resize_param={_has_resize}, eye_dist_param={_has_eye_dist})')

            _net_g_ref        = net_g
            _device_ref       = _device
            _face_helper_ref  = face_helper
            _has_resize_ref   = _has_resize
            _has_eye_dist_ref = _has_eye_dist

            class _GFPGANTorch:
                def enhance(self_, bgr_img, **kw):
                    _face_helper_ref.clean_all()
                    _face_helper_ref.read_image(bgr_img)

                    # call with only the params the installed version accepts
                    _kw = dict(only_center_face=False)
                    if _has_resize_ref:
                        _kw['resize'] = 640
                    if _has_eye_dist_ref:
                        _kw['eye_dist_threshold'] = 5
                    _face_helper_ref.get_face_landmarks_5(**_kw)
                    _face_helper_ref.align_warp_face()

                    if not _face_helper_ref.cropped_faces:
                        # no face found — return original unchanged
                        return None, None, bgr_img

                    for cf in _face_helper_ref.cropped_faces:
                        # cf is BGR (OpenCV/facexlib default) — network trained on RGB
                        rgb_cf = cf[:, :, ::-1].astype(np.float32) / 255.0
                        t = (_torch.from_numpy(rgb_cf)
                             .permute(2, 0, 1).unsqueeze(0).to(_device_ref))
                        t = t * 2 - 1   # [0,1] → [-1, 1]
                        with _torch.no_grad():
                            out = _net_g_ref(t, return_rgb=False,
                                             randomize_noise=True)[0]
                        # out is RGB [-1,1] → uint8, then RGB→BGR for facexlib
                        out = ((out * 0.5 + 0.5).clamp(0, 1)
                               .squeeze(0).permute(1, 2, 0)
                               .cpu().numpy() * 255).astype(np.uint8)
                        _face_helper_ref.add_restored_face(out[:, :, ::-1].copy())

                    _face_helper_ref.get_inverse_affine(None)
                    result = _face_helper_ref.paste_faces_to_input_image()
                    return None, None, result

            _gfpgan_restorer = _GFPGANTorch()
            print('✅  GFPGAN Tier-2 torch inference ready — sharp eyes incoming!')
            return _gfpgan_restorer

    except Exception as e:
        import traceback
        print(f'⚠️  GFPGAN Tier-2 load failed: {type(e).__name__}: {e}')
        traceback.print_exc()

    return None


def _opencv_enhance_face(bgr_frame, face_bbox):
    """
    OpenCV-only face enhancement — fallback when GFPGAN unavailable.
    Multi-stage pipeline on the face ROI:
      1. CLAHE (clipLimit=3)   → strong local contrast — recovers lost detail
      2. Bilateral filter       → smooths skin, keeps hard edges
      3. Unsharp mask × 2      → two-pass sharpening (tight + ultra-tight sigma)
      4. Micro-detail boost     → very tight sigma pass for pore-level texture
    """
    if face_bbox is None:
        return bgr_frame
    x1, y1, x2, y2 = face_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(bgr_frame.shape[1], x2), min(bgr_frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return bgr_frame

    roi = bgr_frame[y1:y2, x1:x2].copy()

    # 1. CLAHE on luminance — stronger clip limit
    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_enh   = clahe.apply(l)
    roi     = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)

    # 2. Bilateral — smooth skin, keep edges
    roi = cv2.bilateralFilter(roi, d=5, sigmaColor=25, sigmaSpace=5)

    # 3. Tight unsharp mask (sigma=0.8) — fine detail
    roi_f = roi.astype(np.float32)
    blur1 = cv2.GaussianBlur(roi_f, (0, 0), 0.8)
    roi_f = np.clip(roi_f + 0.70 * (roi_f - blur1), 0, 255)

    # 4. Ultra-tight unsharp (sigma=0.4) — micro texture
    blur2 = cv2.GaussianBlur(roi_f, (0, 0), 0.4)
    roi_f = np.clip(roi_f + 0.50 * (roi_f - blur2), 0, 255)

    bgr_frame[y1:y2, x1:x2] = roi_f.astype(np.uint8)
    return bgr_frame


def _gfpgan_enhance_frame(restorer, bgr_frame, face_bbox=None):
    """
    Enhance a frame. Uses GFPGAN if available, OpenCV CLAHE+bilateral otherwise.
    """
    if restorer is not None:
        try:
            _, _, enhanced = restorer.enhance(
                bgr_frame, has_aligned=False,
                only_center_face=False, paste_back=True, weight=0.5)
            if enhanced is not None:
                return enhanced
        except Exception as e:
            print(f'  ⚠️  enhancer error: {e}')

    # OpenCV fallback — always works
    return _opencv_enhance_face(bgr_frame, face_bbox)


def enhance_faces(video_path, output_name="enhanced.mp4", weight=0.5):
    """
    Run GFPGAN face enhancement on every frame of a video.
    Works on any video — use as a standalone polish step after face swap.

    weight: 0.0 = very subtle,  0.5 = balanced (recommended),  1.0 = maximum
    Install first: from agent menu → 'Install Face Enhancer (GFPGAN)'
    """
    import time as _time
    t0 = _timed_action("enhance_faces")

    verify_video_path(video_path)
    restorer = _load_gfpgan()
    if restorer is None:
        return ("❌ GFPGAN not ready.\n"
                "Run option 'Install Face Enhancer (GFPGAN)' from the agent menu first.")

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp    = out_path('_tmp_enhanced.mp4')
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    print(f'✨ GFPGAN enhancing {total} frames  (weight={weight}) …')
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _gfpgan_enhance_frame(restorer, frame, face_bbox=None)
        writer.write(frame)
        i += 1
        if i % 20 == 0:
            elapsed = _time.time() - t0
            spf = elapsed / i
            eta = int(spf * max(0, total - i))
            print(f'   frame {i}/{total}  ETA ~{eta//60}m{eta%60:02d}s')

    cap.release()
    writer.release()

    # Mux audio
    out = out_path(output_name)
    has_audio = bool(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True).stdout.strip())
    cmd = ['ffmpeg', '-y', '-i', tmp]
    if has_audio:
        cmd += ['-i', video_path, '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
                '-c:a', 'aac', out]
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '17', out]
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        os.remove(tmp)
    except Exception:
        pass

    _log_timed("enhance_faces", t0, {"weight": weight})
    return done(out)


# ─────────────────────────────────────────────────────────────────────────────
# ④-B  FALLBACK FACE SWAP  (OpenCV Haar + CSRT tracker + seamlessClone)
#       No extra install needed. Less identity-accurate but always available.
# ─────────────────────────────────────────────────────────────────────────────

def _match_color(src, target):
    """Reinhard colour transfer: match src mean/std to target per channel."""
    src_f = src.astype(np.float32)
    tgt_f = target.astype(np.float32)
    for c in range(3):
        s_m, s_s = src_f[:,:,c].mean(), src_f[:,:,c].std() + 1e-6
        t_m, t_s = tgt_f[:,:,c].mean(), tgt_f[:,:,c].std() + 1e-6
        src_f[:,:,c] = (src_f[:,:,c] - s_m) * (t_s / s_s) + t_m
    return np.clip(src_f, 0, 255).astype(np.uint8)


def _crop_source_face(img_bgr):
    """Crop to face region from source photo. Falls back to full image."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray)
    best  = _largest_face(faces)
    if best is None:
        return img_bgr
    x, y, w, h = [int(v) for v in best]
    x1 = max(0, x - int(w*0.22));           y1 = max(0, y - int(h*0.40))
    x2 = min(img_bgr.shape[1], x+w+int(w*0.22))
    y2 = min(img_bgr.shape[0], y+h+int(h*0.12))
    return img_bgr[y1:y2, x1:x2]


def _seamless_swap(frame_bgr, src_face_bgr, face_rect):
    """Paste src_face_bgr over face_rect in frame using Poisson blending."""
    x, y, w, h = [int(v) for v in face_rect]
    fH, fW = frame_bgr.shape[:2]
    px = int(w*0.28); py_top = int(h*0.55); py_bot = int(h*0.18)
    x1 = max(0, x-px);           y1 = max(0, y-py_top)
    x2 = min(fW, x+w+px);        y2 = min(fH, y+h+py_bot)
    tw = x2-x1; th = y2-y1
    if tw <= 0 or th <= 0:
        return frame_bgr

    target_roi  = frame_bgr[y1:y2, x1:x2]
    src_matched = _match_color(src_face_bgr, target_roi)
    src_resized = cv2.resize(src_matched, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    mask = np.zeros((th, tw), dtype=np.uint8)
    cx, cy = tw//2, th//2
    cv2.ellipse(mask, (cx,cy), (max(1,cx-6), max(1,cy-6)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 15)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    center = (x1+cx, y1+cy)
    margin = 6
    if (center[0] < margin or center[0] > fW-margin or
            center[1] < margin or center[1] > fH-margin):
        return _alpha_blend_swap(frame_bgr, src_resized, mask, x1, y1)
    try:
        return cv2.seamlessClone(src_resized, frame_bgr, mask, center, cv2.MIXED_CLONE)
    except cv2.error:
        return _alpha_blend_swap(frame_bgr, src_resized, mask, x1, y1)


def _alpha_blend_swap(frame_bgr, src_bgr, mask_gray, x1, y1):
    result = frame_bgr.copy()
    th, tw = src_bgr.shape[:2]
    fH, fW = frame_bgr.shape[:2]
    y2 = min(fH, y1+th); x2 = min(fW, x1+tw)
    sh = y2-y1; sw = x2-x1
    if sh <= 0 or sw <= 0:
        return frame_bgr
    m = mask_gray[:sh,:sw].astype(np.float32)/255.0
    m3 = cv2.merge([m,m,m])
    roi = result[y1:y2, x1:x2].astype(np.float32)
    src_c = src_bgr[:sh,:sw].astype(np.float32)
    result[y1:y2, x1:x2] = np.clip(m3*src_c + (1-m3)*roi, 0, 255).astype(np.uint8)
    return result


def _face_swap_opencv_tracked(video_path, face_image_path, output_name):
    """
    Fallback face swap: Haar detection on first frame → CSRT tracker for all
    subsequent frames (tracker NEVER loses the face between detections).
    """
    src_bgr_full = cv2.imread(face_image_path)
    if src_bgr_full is None:
        src_bgr_full = cv2.cvtColor(
            np.array(Image.open(face_image_path).convert('RGB')),
            cv2.COLOR_RGB2BGR)
    src_face_bgr = _crop_source_face(src_bgr_full)

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_video = out_path("_tmp_faceswap_video.mp4")
    writer    = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    tracker      = None
    face_bbox    = None   # (x, y, w, h)
    detect_every = max(1, int(fps))    # re-detect once per second

    print(f"🔄 Processing {total} frames (tracked OpenCV fallback) …")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Re-detect every `detect_every` frames ──────────────────
        if frame_idx % detect_every == 0 or face_bbox is None:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces(gray)
            best  = _largest_face(faces)
            if best is not None:
                face_bbox = tuple(int(v) for v in best)
                # Re-init CSRT tracker on fresh detection
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, face_bbox)

        elif tracker is not None:
            # ── Update tracker between detections ──────────────────
            ok, box = tracker.update(frame)
            if ok:
                face_bbox = tuple(int(v) for v in box)
            # If tracker fails, keep last known bbox until next detection

        # ── Apply swap if we have any face position ─────────────────
        if face_bbox is not None:
            frame = _seamless_swap(frame, src_face_bgr, face_bbox)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-mux audio
    out = out_path(output_name)
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True)
    has_audio = bool(probe.stdout.strip())
    if has_audio:
        subprocess.run([
            'ffmpeg', '-y', '-i', tmp_video, '-i', video_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-shortest', out
        ], check=True, capture_output=True)
    else:
        subprocess.run([
            'ffmpeg', '-y', '-i', tmp_video,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', out
        ], check=True, capture_output=True)
    try:
        os.remove(tmp_video)
    except Exception:
        pass
    return done(out)


# ─────────────────────────────────────────────────────────────────────────────
# ④-D  REPLICATE CLOUD FACE SWAP
#       Sends frames to Replicate API — no local GPU/model needed.
#       Quality: production-grade (better than local inswapper_128).
#       Cost: ~$0.003/frame  →  ~$0.10–0.50 per typical short video.
#       Model: codeplugtech/face-swap  (fast, sharp, great skin tone)
# ─────────────────────────────────────────────────────────────────────────────

_REPLICATE_MODEL    = "arabyai-replicate/roop_face_swap"   # kept for reference
_REPLICATE_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "replicate_key.txt")
_FAL_KEY_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "fal_key.txt")


def _save_replicate_key(api_key: str):
    with open(_REPLICATE_KEY_FILE, 'w') as f:
        f.write(api_key.strip())


def _load_replicate_key() -> str | None:
    if os.path.exists(_REPLICATE_KEY_FILE):
        with open(_REPLICATE_KEY_FILE) as f:
            k = f.read().strip()
            return k if k else None
    return None


def face_swap_replicate(video_path, face_source, api_key=None,
                        output_name=None, swap_every=2):
    """
    Cloud face swap via Replicate API.
    Model: easel/advanced-face-swap (confirmed live April 2026)
    Processes key frames as images, reassembles to video.
    With $5+ credit: parallel, fast. Under $5: sequential with backoff.
    """
    t0 = _timed_action("face_swap_replicate")

    verify_video_path(video_path)
    face_image_path, _tmp = _resolve_face_image(face_source)
    if not os.path.exists(face_image_path):
        return f"❌ Face image not found: {face_image_path}"

    key = api_key or _load_replicate_key()
    if not key:
        return ("❌ No Replicate API key found.\n"
                "Run option 22 to set your key first.\n"
                "Get it at: https://replicate.com/account/api-tokens")

    try:
        import replicate as _rep
    except ImportError:
        subprocess.run(['pip3', 'install', 'replicate', '--break-system-packages', '-q'],
                       capture_output=True)
        import replicate as _rep

    client = _rep.Client(api_token=key)

    if output_name is None:
        output_name = f"face_swapped_replicate_{datetime.now().strftime('%H%M%S')}.mp4"
    out = out_path(output_name)

    # ── read video frames ────────────────────────────────────────────
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    secs   = total / fps if fps else 0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    key_indices = list(range(0, len(frames), swap_every))
    est_cost    = len(key_indices) * 0.0023
    print(f"📹 {total} frames  {secs:.1f}s  @ {fps:.0f}fps")
    print(f"☁️  Model: easel/advanced-face-swap  ({len(key_indices)} API calls  ~${est_cost:.2f})")

    import urllib.request
    tmp_dir = out_path("_rep_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── resolve model version once (avoids 422 "version not found") ──
    _model_id = "easel/advanced-face-swap"
    try:
        _model   = client.models.get(_model_id)
        _version = _model.latest_version
        print(f"   Model version: {str(_version.id)[:16]}…")
        use_version = True
    except Exception as ve:
        print(f"   ⚠️  Could not fetch version ({ve}) — trying direct run")
        use_version = False

    def _call_one(idx, frame, retries=6):
        tmp_f = os.path.join(tmp_dir, f"f{idx:06d}.jpg")
        cv2.imwrite(tmp_f, frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        wait = 12
        for attempt in range(retries):
            try:
                with open(tmp_f, 'rb') as tf, open(face_image_path, 'rb') as ff:
                    inp = {"target_image": tf, "swap_image": ff}
                    if use_version:
                        result = _version.predict(**inp)
                    else:
                        result = client.run(_model_id, input=inp)
                # result may be a URL string or a list
                url = result[0] if isinstance(result, list) else str(result)
                with urllib.request.urlopen(url) as r:
                    data = r.read()
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
                try: os.remove(tmp_f)
                except: pass
                return idx, img
            except Exception as e:
                err = str(e)
                if '429' in err or 'throttled' in err.lower():
                    if attempt < retries - 1:
                        print(f"   ⏳ Rate limit hit — waiting {wait}s …")
                        time.sleep(wait); wait = min(wait * 2, 60)
                        continue
                if '422' in err or 'permission' in err.lower():
                    # Need to accept terms on Replicate website first
                    try: os.remove(tmp_f)
                    except: pass
                    raise RuntimeError(
                        f"❌ Model requires terms acceptance.\n"
                        f"   Open this URL in your browser and click Agree:\n"
                        f"   https://replicate.com/{_model_id}\n"
                        f"   Then run option 21 again.")
                print(f"   ⚠️  frame {idx}: {e}")
                try: os.remove(tmp_f)
                except: pass
                return idx, frame
        return idx, frame

    swapped = {}
    done    = 0
    for idx in key_indices:
        try:
            ri, rf = _call_one(idx, frames[idx])
        except RuntimeError as re:
            return str(re)   # terms-acceptance message — stop immediately
        swapped[ri] = rf
        done += 1
        if done % 5 == 0 or done == len(key_indices):
            print(f"   ☁️  {done}/{len(key_indices)} frames  ({done/len(key_indices)*100:.0f}%)")
        if done < len(key_indices):
            time.sleep(0.3)   # small gap — avoids hammering API

    # ── assemble video ───────────────────────────────────────────────
    tmp_vid = out_path("_tmp_rep.mp4")
    wrt     = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    last    = None
    for i, frm in enumerate(frames):
        if i in swapped and swapped[i] is not None:
            last = swapped[i]
        wrt.write(last if last is not None else frm)
    wrt.release()

    has_audio = bool(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True).stdout.strip())
    cmd = ['ffmpeg', '-y', '-i', tmp_vid]
    if has_audio:
        cmd += ['-i', video_path, '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-c:a', 'aac', out]
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '16', out]
    subprocess.run(cmd, check=True, capture_output=True)

    try:
        os.remove(tmp_vid)
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
    except: pass

    actual_cost = len(swapped) * 0.0023
    _log_timed("face_swap_replicate", t0,
               {"engine": "replicate/easel", "api_calls": len(swapped),
                "est_cost_usd": round(actual_cost, 3)})
    return (f"✅ Replicate face swap done!\n"
            f"   Frames swapped: {len(swapped)}/{len(key_indices)}\n"
            f"   Actual cost: ~${actual_cost:.2f}\n"
            f"   Output: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ④  MAIN FACE SWAP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def face_swap(video_path, face_source, output_name=None,
              openai_api_key=None, target_face='largest'):
    """
    Replace a face in the video with face_source.

    face_source: image path | @instagram | description.json | https://url

    target_face: which face to swap when multiple people are in the frame
      'largest'  – biggest face (default — the main subject)
      'left'     – leftmost person
      'right'    – rightmost person
      'center'   – person closest to the frame centre
      'top'      – person highest in frame
      'bottom'   – person lowest in frame
      '1' / '2' / '3'  – rank by size (1 = largest)

    Engine priority:
      1. InsightFace + inswapper_128  (Kling-level quality, requires install)
         pip3 install insightface onnxruntime --break-system-packages
      2. OpenCV CSRT tracker + Poisson blend  (always available, good quality)
    """
    t0 = _timed_action("face_swap")
    # Always generate a unique timestamped filename so every run is saved
    if output_name is None:
        output_name = f"face_swapped_{datetime.now().strftime('%H%M%S')}.mp4"
    verify_video_path(video_path)

    try:
        face_image_path, tmp_to_delete = _resolve_face_image(
            face_source, openai_api_key=openai_api_key)
    except Exception as e:
        log_video_action("face_swap", success=False, error=f"resolve_face_image: {e}")
        return f"❌ Could not resolve face source: {e}"

    try:
        # ── Try deep-learning engine first ──────────────────────────
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401
            print("✅ InsightFace detected — using deep-learning face swap")
            result = _face_swap_insightface(video_path, face_image_path, output_name)
            _log_timed("face_swap", t0, {"engine": "insightface", "source_type": type(face_source).__name__})
            return result
        except ImportError:
            print("ℹ️  InsightFace not installed — using OpenCV tracked fallback")
            print("   For Kling-level quality run:")
            print("   pip3 install insightface onnxruntime --break-system-packages")

        # ── Fallback: CSRT tracked + Poisson blend ──────────────────
        result = _face_swap_opencv_tracked(video_path, face_image_path, output_name)
        _log_timed("face_swap", t0, {"engine": "opencv_tracked"})
        return result

    except Exception as e:
        _log_timed("face_swap", t0, {}, success=False, error=str(e))
        raise
    finally:
        if tmp_to_delete and os.path.exists(tmp_to_delete):
            try:
                os.remove(tmp_to_delete)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ COLOR GRADING
# ─────────────────────────────────────────────────────────────────────────────

def color_grade(video_path, preset="cinematic", output_name=None):
    """
    Apply a color grading preset.
    Presets: cinematic, warm, cool, vintage, bw, vivid, muted, neon, golden, teal_orange
    """
    if not output_name:
        output_name = f"{preset}_{os.path.basename(video_path)}"
    p = COLOR_PRESETS.get(preset, COLOR_PRESETS["cinematic"])
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)

    def grade(get_frame, t):
        f = get_frame(t).astype(np.float32)
        if p.get('bw'):
            g = np.mean(f, axis=2, keepdims=True)
            return np.clip(np.repeat(g, 3, axis=2), 0, 255).astype(np.uint8)
        # Brightness
        f *= p.get('brightness', 1.0)
        # Contrast
        f = (f - 128) * p.get('contrast', 1.0) + 128
        # Saturation
        gray = np.mean(f, axis=2, keepdims=True)
        f = gray + p.get('saturation', 1.0) * (f - gray)
        # Color tint
        if 'shadow' in p or 'highlight' in p:
            lum = np.clip(np.mean(f, axis=2, keepdims=True) / 255.0, 0, 1)
            sh = np.array(p.get('shadow', (0, 0, 0)), dtype=np.float32)
            hi = np.array(p.get('highlight', (255, 255, 255)), dtype=np.float32)
            tint = (1 - lum) * sh + lum * hi
            f = f * 0.85 + tint * 0.15
        return np.clip(f, 0, 255).astype(np.uint8)

    out = out_path(output_name)
    clip.fl(grade).write_videofile(out, fps=clip.fps, logger=None)
    log_video_action("color_grade", {"preset": preset})
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ EFFECTS & OVERLAYS
# ─────────────────────────────────────────────────────────────────────────────

def add_watermark(video_path, text, position='bottom-right',
                  fontsize=32, opacity=0.75, output_name="watermarked.mp4"):
    """Add a text watermark at a corner of the video."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    W, H = clip.size
    font = _get_font(fontsize)
    dummy = Image.new('RGBA', (W, H))
    draw = ImageDraw.Draw(dummy)
    tw, th = _text_size(draw, text, font)
    pad = 20

    pos_map = {
        'bottom-right': (W - tw - pad, H - th - pad),
        'bottom-left':  (pad, H - th - pad),
        'top-right':    (W - tw - pad, pad),
        'top-left':     (pad, pad),
        'center':       ((W - tw) // 2, (H - th) // 2),
    }
    x, y = pos_map.get(position, pos_map['bottom-right'])

    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, int(opacity * 150)))
    d.text((x, y), text, font=font, fill=(255, 255, 255, int(opacity * 255)))

    arr = np.array(img)
    wm = ImageClip(arr[:, :, :3]).set_duration(clip.duration)
    wm = wm.set_mask(ImageClip(arr[:, :, 3] / 255.0, ismask=True))
    final = CompositeVideoClip([clip, wm])
    out = out_path(output_name)
    final.write_videofile(out, fps=clip.fps, logger=None)
    log_video_action("add_watermark", {"position": position, "fontsize": fontsize})
    return done(out)

def add_lower_third(video_path, name, title="",
                    start=0, end=None, accent_color=(20, 100, 220),
                    output_name="lower_third.mp4"):
    """Add a professional news-style lower third graphic."""
    clip = VideoFileClip(video_path)
    W, H = clip.size
    dur = clip.duration
    end = end if end is not None else dur

    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bar_y = int(H * 0.72)
    bar_h = int(H * 0.13)
    bar_w = int(W * 0.62)

    r, g, b = accent_color
    draw.rectangle([0, bar_y, bar_w, bar_y + bar_h], fill=(r, g, b, 220))
    draw.rectangle([0, bar_y + bar_h, bar_w, bar_y + bar_h + 7],
                   fill=(255, 200, 0, 255))

    name_font = _get_font(int(bar_h * 0.48))
    title_font = _get_font(int(bar_h * 0.32))
    draw.text((24, bar_y + 10), name, font=name_font, fill='white')
    if title:
        draw.text((24, bar_y + bar_h - int(bar_h * 0.38) - 8), title,
                  font=title_font, fill=(255, 230, 100))

    arr = np.array(img)
    lt = ImageClip(arr[:, :, :3]).set_mask(
        ImageClip(arr[:, :, 3] / 255.0, ismask=True))
    lt = lt.set_start(start).set_end(min(end, dur)).fadein(0.4).fadeout(0.4)

    out = out_path(output_name)
    CompositeVideoClip([clip, lt]).write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def picture_in_picture(main_video, pip_video, position='bottom-right',
                        scale=0.28, output_name="pip.mp4"):
    """Overlay a smaller video on top of the main video."""
    main = VideoFileClip(main_video)
    pip = VideoFileClip(pip_video)
    W, H = main.size
    pip_w = int(W * scale)
    pip_h = int(pip_w * pip.h / pip.w)
    pad = 20
    pos_map = {
        'bottom-right': (W - pip_w - pad, H - pip_h - pad),
        'bottom-left':  (pad, H - pip_h - pad),
        'top-right':    (W - pip_w - pad, pad),
        'top-left':     (pad, pad),
    }
    x, y = pos_map.get(position, pos_map['bottom-right'])
    pip_clip = pip.resize((pip_w, pip_h))
    if pip_clip.duration > main.duration:
        pip_clip = pip_clip.subclip(0, main.duration)
    pip_clip = pip_clip.set_position((x, y))
    out = out_path(output_name)
    CompositeVideoClip([main, pip_clip]).write_videofile(out, fps=main.fps, logger=None)
    return done(out)

def add_subtitles_srt(video_path, srt_path, fontsize=40,
                      output_name="subtitled.mp4"):
    """Burn in subtitles from an .srt file."""
    def parse_srt(path):
        with open(path, encoding='utf-8') as f:
            content = f.read()
        subs = []
        for block in content.strip().split('\n\n'):
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            ts = lines[1].split(' --> ')
            def to_sec(s):
                s = s.strip().replace(',', '.')
                h, m, sec = s.split(':')
                return int(h) * 3600 + int(m) * 60 + float(sec)
            subs.append((to_sec(ts[0]), to_sec(ts[1]), ' '.join(lines[2:])))
        return subs

    clip = VideoFileClip(video_path)
    txt_clips = []
    for start, end, text in parse_srt(srt_path):
        if start >= clip.duration:
            continue
        tc = make_text_overlay(text, clip.size, fontsize=fontsize,
                               color='white', stroke_color='black',
                               stroke_width=2, bg_alpha=0.6, position='bottom')
        tc = tc.set_start(start).set_end(min(end, clip.duration))
        txt_clips.append(tc)

    out = out_path(output_name)
    CompositeVideoClip([clip] + txt_clips).write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

def add_transition(video1_path, video2_path, transition="crossfade",
                   duration=1.0, output_name="with_transition.mp4"):
    """
    Join two videos with a transition.
    transition: crossfade | fade | wipe_left
    """
    c1 = VideoFileClip(video1_path)
    c2 = VideoFileClip(video2_path).resize(c1.size)

    if transition == "crossfade":
        c1_out = c1.fadeout(duration)
        c2_in = c2.fadein(duration).set_start(c1.duration - duration)
        total = c1.duration + c2.duration - duration
        final = CompositeVideoClip([c1_out, c2_in]).subclip(0, total)
    elif transition == "fade":
        final = concatenate_videoclips([c1.fadeout(duration), c2.fadein(duration)],
                                       method='compose')
    else:
        final = concatenate_videoclips([c1, c2], method='compose')

    out = out_path(output_name)
    final.write_videofile(out, fps=c1.fps, logger=None)
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ⑦ KEN BURNS & ZOOM
# ─────────────────────────────────────────────────────────────────────────────

def _ken_burns_clip(image_path, duration, fps=30,
                    zoom_start=1.0, zoom_end=1.28, target_size=(1920, 1080)):
    """Internal: Ken Burns animated clip from an image."""
    W, H = target_size
    img = Image.open(image_path).convert('RGB')
    r = img.width / img.height
    if r > W / H:
        nw, nh = int(H * r), H
    else:
        nw, nh = W, int(W / r)
    img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.array(img)

    def make_frame(t):
        prog = t / max(duration, 0.001)
        zoom = zoom_start + (zoom_end - zoom_start) * prog
        pw, ph = int(W / zoom), int(H / zoom)
        x1 = max(0, min((nw - pw) // 2 + int(nw * 0.02 * prog), nw - pw))
        y1 = max(0, min((nh - ph) // 2 + int(nh * 0.02 * prog), nh - ph))
        cropped = arr[y1:y1 + ph, x1:x1 + pw]
        return np.array(Image.fromarray(cropped).resize((W, H), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).set_fps(fps)

def ken_burns(image_path, duration=5, zoom_start=1.0, zoom_end=1.3,
              output_name="ken_burns.mp4", fps=30):
    """Cinematic pan-and-zoom on a still photo."""
    clip = _ken_burns_clip(image_path, duration, fps, zoom_start, zoom_end)
    clip = clip.fadein(0.5).fadeout(0.5)
    out = out_path(output_name)
    clip.write_videofile(out, fps=fps, logger=None)
    return done(out)

def zoom_effect(video_path, zoom_start=1.0, zoom_end=1.5,
                output_name="zoomed.mp4"):
    """Progressive zoom in or out during the entire clip."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)

    def zoom_frame(get_frame, t):
        frame = get_frame(t)
        prog = t / max(clip.duration, 0.001)
        zoom = zoom_start + (zoom_end - zoom_start) * prog
        H, W = frame.shape[:2]
        nw, nh = int(W / zoom), int(H / zoom)
        x1, y1 = (W - nw) // 2, (H - nh) // 2
        cropped = frame[y1:y1 + nh, x1:x1 + nw]
        return cv2.resize(cropped, (W, H))

    out = out_path(output_name)
    clip.fl(zoom_frame).write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ⑧ AI VOICEOVER & SCRIPT-TO-VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def ai_voiceover(script_text, voice='Samantha', rate=175,
                 output_name="voiceover.mp3"):
    """
    Generate AI voiceover from text using macOS TTS (no API key needed).
    Popular voices: Samantha, Alex, Victoria, Karen, Daniel, Moira, Tessa
    """
    out = out_path(output_name)
    aiff = out.replace('.mp3', '.aiff')
    subprocess.run(['say', '-v', voice, '-r', str(rate), '-o', aiff, script_text],
                   check=True)
    subprocess.run(['ffmpeg', '-y', '-i', aiff, out], capture_output=True)
    try:
        os.remove(aiff)
    except Exception:
        pass
    return done(out)

def _gradient_frame(W, H, index=0):
    """Create a gradient background numpy array."""
    palettes = [
        ((15, 15, 50), (80, 30, 110)),
        ((15, 45, 85), (0, 110, 150)),
        ((50, 15, 15), (110, 50, 0)),
        ((15, 50, 15), (30, 100, 60)),
        ((10, 10, 10), (50, 50, 50)),
    ]
    c1, c2 = palettes[index % len(palettes)]
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        t = i / H
        arr[i, :] = [int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3)]
    return arr

def script_to_video(script, images_folder=None, voice='Samantha',
                    fps=30, output_name="script_video.mp4"):
    """
    Full pipeline: text script → AI voiceover → captions → video.

    script can be:
      - A plain string  (split into sentences automatically)
      - A list of dicts: [{"text": "...", "image": "optional_path.jpg"}, ...]

    images_folder: optional folder of images to use as backgrounds (Ken Burns)
    """
    # Normalise script
    if isinstance(script, str):
        sentences = re.split(r'(?<=[.!?])\s+', script.strip())
        segments = [{"text": s.strip()} for s in sentences if s.strip()]
    else:
        segments = [s for s in script if s.get("text", "").strip()]

    if not segments:
        return "❌ No script content provided."

    # Gather images if folder given
    folder_images = []
    if images_folder and os.path.isdir(images_folder):
        folder_images = sorted([
            os.path.join(images_folder, f)
            for f in os.listdir(images_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])

    clips = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Generate voiceover
        aiff = out_path(f"_seg_{i}.aiff")
        subprocess.run(['say', '-v', voice, '-r', '165', '-o', aiff, text], check=True)
        audio = AudioFileClip(aiff)
        dur = audio.duration + 0.25

        # Background
        img_path = seg.get("image")
        if img_path and os.path.exists(img_path):
            bg = _ken_burns_clip(img_path, dur, fps)
        elif folder_images:
            bg = _ken_burns_clip(folder_images[i % len(folder_images)], dur, fps)
        else:
            bg_arr = _gradient_frame(1920, 1080, i)
            bg = ImageClip(bg_arr, duration=dur).set_fps(fps)

        # Caption
        cap = make_text_overlay(text, (1920, 1080), fontsize=46,
                                color='white', stroke_color='black',
                                stroke_width=2, bg_alpha=0.6,
                                position='bottom', duration=dur)
        seg_clip = CompositeVideoClip([bg, cap]).set_audio(audio)
        seg_clip = seg_clip.fadein(0.2).fadeout(0.2)
        clips.append(seg_clip)
        try:
            os.remove(aiff)
        except Exception:
            pass

    if not clips:
        return "❌ Nothing to render."

    final = concatenate_videoclips(clips, method='compose')
    out = out_path(output_name)
    final.write_videofile(out, fps=fps, logger=None)
    log_video_action("script_to_video", {"voice": voice, "segments": len(segments), "fps": fps})
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ⑨ AI AVATAR VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def create_ai_avatar(script_text, avatar_image_path, voice='Samantha',
                     title="", subtitle="", platform="portrait",
                     output_name="avatar_video.mp4"):
    """
    Create a talking avatar video:
    - Your photo shown in a circular frame
    - AI voiceover narration (macOS TTS)
    - Synchronized word-chunk captions
    - Professional lower third with name/title
    - Gradient background

    For realistic lip-sync: install Wav2Lip (advanced, GPU recommended)
    """
    W, H = PLATFORM_SIZES.get(platform, (1080, 1920))

    # Voiceover
    aiff = out_path("_avatar_voice.aiff")
    subprocess.run(['say', '-v', voice, '-r', '170', '-o', aiff, script_text], check=True)
    audio = AudioFileClip(aiff)
    dur = audio.duration

    # Gradient background
    bg_arr = _gradient_frame(W, H, 0)
    bg = ImageClip(bg_arr, duration=dur).set_fps(30)
    layers = [bg]

    # Avatar circle
    if os.path.exists(avatar_image_path):
        av_img = Image.open(avatar_image_path).convert('RGBA')
        # Square-crop from center
        side = min(av_img.size)
        l = (av_img.width - side) // 2
        t = (av_img.height - side) // 2
        av_img = av_img.crop((l, t, l + side, t + side))
        av_size = int(W * 0.62)
        av_img = av_img.resize((av_size, av_size), Image.LANCZOS)
        # Circular mask
        msk = Image.new('L', (av_size, av_size), 0)
        ImageDraw.Draw(msk).ellipse([0, 0, av_size, av_size], fill=255)
        av_img.putalpha(msk)
        # Ring border
        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ax, ay = (W - av_size) // 2, int(H * 0.08)
        # Draw accent ring
        ring_d = ImageDraw.Draw(overlay)
        ring_d.ellipse([ax - 8, ay - 8, ax + av_size + 8, ay + av_size + 8],
                       outline=(255, 200, 50, 220), width=8)
        overlay.paste(av_img, (ax, ay), av_img)
        arr = np.array(overlay)
        av_clip = ImageClip(arr[:, :, :3]).set_duration(dur)
        av_clip = av_clip.set_mask(ImageClip(arr[:, :, 3] / 255.0, ismask=True))
        layers.append(av_clip)

    # Lower third
    if title:
        lt_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        lt_d = ImageDraw.Draw(lt_img)
        bar_y = int(H * 0.73)
        bar_h = int(H * 0.10)
        lt_d.rectangle([0, bar_y, W, bar_y + bar_h], fill=(20, 100, 220, 220))
        lt_d.rectangle([0, bar_y + bar_h, W, bar_y + bar_h + 7], fill=(255, 200, 0, 255))
        lt_d.text((28, bar_y + 12), title, font=_get_font(int(bar_h * 0.45)), fill='white')
        if subtitle:
            lt_d.text((28, bar_y + bar_h - int(bar_h * 0.42) - 8), subtitle,
                      font=_get_font(int(bar_h * 0.3)), fill=(255, 230, 100))
        lt_arr = np.array(lt_img)
        lt_clip = ImageClip(lt_arr[:, :, :3]).set_duration(dur)
        lt_clip = lt_clip.set_mask(ImageClip(lt_arr[:, :, 3] / 255.0, ismask=True))
        lt_clip = lt_clip.fadein(0.5)
        layers.append(lt_clip)

    # Word-chunk captions (synced to approximate timing)
    words = script_text.split()
    chunk_size = 6
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunk_dur = dur / max(len(chunks), 1)
    for i, chunk in enumerate(chunks):
        tc = make_text_overlay(chunk, (W, H), fontsize=36,
                               color='white', stroke_color='black',
                               stroke_width=2, bg_alpha=0.0, position='bottom')
        tc = tc.set_start(i * chunk_dur).set_end((i + 1) * chunk_dur)
        layers.append(tc)

    final = CompositeVideoClip(layers).set_audio(audio).fadein(0.3).fadeout(0.3)
    out = out_path(output_name)
    final.write_videofile(out, fps=30, logger=None)
    log_video_action("create_ai_avatar", {"voice": voice, "script_words": len(script_text.split())})
    try:
        os.remove(aiff)
    except Exception:
        pass
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# ⑩ SLIDESHOW & FRAME TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def images_to_video(images_folder, duration_per_image=3, fps=30,
                    ken_burns_effect=True, narration_path=None,
                    output_name="slideshow.mp4"):
    """Create a video from a folder of images, with optional Ken Burns and narration."""
    imgs = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    if not imgs:
        return "❌ No images found in folder."

    clips = []
    for p in imgs:
        if ken_burns_effect:
            c = _ken_burns_clip(p, duration_per_image, fps)
        else:
            arr = np.array(Image.open(p).convert('RGB').resize((1920, 1080), Image.LANCZOS))
            c = ImageClip(arr, duration=duration_per_image).set_fps(fps)
        clips.append(c.fadein(0.3).fadeout(0.3))

    final = concatenate_videoclips(clips, method='compose')
    if narration_path and os.path.exists(narration_path):
        audio = AudioFileClip(narration_path)
        if audio.duration > final.duration:
            audio = audio.subclip(0, final.duration)
        final = final.set_audio(audio)

    out = out_path(output_name)
    final.write_videofile(out, fps=fps, logger=None)
    return done(out)

def extract_frames(video_path, every_n_seconds=1, output_folder=None):
    """Extract frames from video as PNG images into a folder."""
    clip = VideoFileClip(video_path)
    if not output_folder:
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = out_path(f"frames_{name}")
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for t in np.arange(0, clip.duration, every_n_seconds):
        Image.fromarray(clip.get_frame(t)).save(
            os.path.join(output_folder, f"frame_{count:04d}.png"))
        count += 1
    subprocess.Popen(["open", output_folder])
    return f"✅ Extracted {count} frames to {output_folder}"

# ─────────────────────────────────────────────────────────────────────────────
# ⑪ VIDEO STABILIZATION
# ─────────────────────────────────────────────────────────────────────────────

def stabilize_video(video_path, smoothing=30, output_name="stabilized.mp4"):
    """Basic optical-flow video stabilization using OpenCV."""
    verify_video_path(video_path)
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames())
    if len(frames) < 2:
        return "❌ Video too short to stabilize."

    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    transforms = []
    for frame in frames[1:]:
        curr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 30)
        identity = np.eye(2, 3, dtype=np.float32)
        if pts is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
            # status and new_pts can be None when optical flow completely fails
            if status is not None and new_pts is not None:
                status_mask = status.ravel() == 1      # boolean array, never ambiguous
                good_old = pts[status_mask]
                good_new = new_pts[status_mask]
                if len(good_old) >= 4:
                    m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                    # m is None or a 2x3 ndarray — use identity check, NOT truth test
                    transforms.append(m if m is not None else identity)
                else:
                    transforms.append(identity)
            else:
                transforms.append(identity)
        else:
            transforms.append(identity)
        prev = curr

    tx = np.array([t[0, 2] for t in transforms])
    ty = np.array([t[1, 2] for t in transforms])
    kernel = np.ones(smoothing) / smoothing
    stx = np.convolve(np.cumsum(tx), kernel, 'same') - np.cumsum(tx)
    sty = np.convolve(np.cumsum(ty), kernel, 'same') - np.cumsum(ty)

    H_s, W_s = frames[0].shape[:2]
    stabilized = [frames[0]]
    for i, frame in enumerate(frames[1:]):
        m = transforms[i].copy()
        m[0, 2] += stx[min(i, len(stx) - 1)]
        m[1, 2] += sty[min(i, len(sty) - 1)]
        stabilized.append(cv2.warpAffine(frame, m, (W_s, H_s),
                                          borderMode=cv2.BORDER_REFLECT))

    def make_frame(t):
        return stabilized[min(int(t * clip.fps), len(stabilized) - 1)]

    result = VideoClip(make_frame, duration=clip.duration).set_fps(clip.fps)
    if clip.audio is not None:
        result = result.set_audio(clip.audio)
    out = out_path(output_name)
    result.write_videofile(out, fps=clip.fps, logger=None)
    return done(out)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CATALOGUE (for menu display)
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = {
    # Basic
    "1":  ("Add Captions (PIL)",            "add_captions"),
    "2":  ("Auto Captions (Whisper AI)",    "auto_captions"),
    "3":  ("Trim Video",                     "trim_video"),
    "4":  ("Merge Videos",                   "merge_videos"),
    "5":  ("Add Background Music",           "add_background_music"),
    "6":  ("Extract Audio",                  "extract_audio"),
    "7":  ("Change Speed",                   "change_speed"),
    "8":  ("Add Fade In/Out",               "add_fade"),
    "9":  ("Add Intro / Outro",             "add_intro_outro"),
    "10": ("Resize for Platform",            "resize_for_platform"),
    # Background
    "11": ("Remove BG from Image (AI)",     "remove_background_image"),
    "12": ("Change Video Background (AI)",  "change_background"),
    "13": ("Chroma Key / Green Screen",     "chroma_key"),
    "14": ("Portrait Background Blur",      "background_blur"),
    # Face
    "15": ("Face Blur",                      "face_blur"),
    "16": ("Face Pixelate",                  "face_pixelate"),
    "17": ("Face Swap",                      "face_swap"),
    # Styling
    "18": ("Color Grade",                    "color_grade"),
    "19": ("Add Watermark",                  "add_watermark"),
    "20": ("Lower Third (News Style)",       "add_lower_third"),
    "21": ("Picture-in-Picture",             "picture_in_picture"),
    "22": ("Add SRT Subtitles",             "add_subtitles_srt"),
    "23": ("Add Transition",                 "add_transition"),
    "24": ("Zoom Effect",                    "zoom_effect"),
    # AI Creation
    "25": ("Ken Burns (Photo → Video)",     "ken_burns"),
    "26": ("AI Voiceover",                   "ai_voiceover"),
    "27": ("Script → Video Pipeline",       "script_to_video"),
    "28": ("AI Avatar Video",               "create_ai_avatar"),
    # Utility
    "29": ("Slideshow from Images",         "images_to_video"),
    "30": ("Extract Frames",                "extract_frames"),
    "31": ("Stabilize Video",               "stabilize_video"),
}

MENU_TEXT = "\n".join(
    f"{'─'*40}\n{'BASIC EDITING' if n=='1' else 'BACKGROUND AI' if n=='11' else 'FACE FEATURES' if n=='15' else 'STYLING & EFFECTS' if n=='18' else 'AI CREATION' if n=='25' else 'UTILITIES' if n=='29' else ''}"
    + (f"\n{'─'*40}" if n in ('1','11','15','18','25','29') else "")
    + f"\n{n}. {name}"
    for n, (name, _) in FEATURES.items()
)
