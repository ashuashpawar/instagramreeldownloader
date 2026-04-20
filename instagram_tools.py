from instagrapi import Client
import time

SESSION_FILE = "/Users/ashwinipawar/claude-agent/instagram_tejalsonavne70.json"
USERNAME = "tejalsonavne70"

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Client()
        _client.load_settings(SESSION_FILE)
    return _client

def search_profile(username):
    try:
        import time
        _client = None
        cl = Client()
        cl.load_settings(SESSION_FILE)
        cl.request_timeout = 30
        time.sleep(1)
        try:
            user = cl.user_info_by_username(username)
            results = [user]
        except Exception as e1:
            print(f"Direct lookup failed: {e1}")
            results = cl.search_users(username)
        print(f"DEBUG: found {len(results)} results for {username}")
        if not results:
            return "No profiles found."
        lines = []
        for user in results[:5]:
            lines.append(f"Username: {user.username}\nName: {user.full_name}\nID: {user.pk}\n---")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

def download_reel(url):
    try:
        cl = get_client()
        media_pk = cl.media_pk_from_url(url)
        path = cl.video_download(media_pk, "/Users/ashwinipawar/Downloads/")
        return f"Downloaded to: {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def post_reel(video_path, caption):
    try:
        cl = get_client()
        cl.clip_upload(video_path, caption)
        return "Reel posted successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

def download_reels_from_user(username, count=10):
    try:
        import time
        cl = Client()
        cl.load_settings(SESSION_FILE)
        cl.request_timeout = 30
        time.sleep(1)
        user = cl.user_info_by_username(username)
        user_id = user.pk
        medias = cl.user_clips(user_id, amount=count)
        if not medias:
            return f"No reels found for {username}"
        downloaded = []
        for i, media in enumerate(medias):
            path = cl.video_download(media.pk, "/Users/ashwinipawar/Downloads/")
            downloaded.append(str(path))
            print(f"Downloaded {i+1}/{len(medias)}: {path}")
            time.sleep(2)
        return f"Downloaded {len(downloaded)} reels to Downloads folder:\n" + "\n".join(downloaded)
    except Exception as e:
        return f"Error: {str(e)}"
