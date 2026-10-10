"""Publish finished videos to YouTube (YouTube Data API v3, videos.insert).

Everything lives in the shared app folder, under youtube/:
  client_secret.json  the OAuth "Desktop app" client Tom downloads from Google Cloud (uploaded on the page)
  token.json          the sign-in, saved after the one-time Google consent screen
  youtube.json        the shared description, the last visibility picked, and what has been published (by file + size + mtime,
                      so a rebuilt reel counts as new)

Gotcha: until Google audits the API project, YouTube locks every API upload as Private, whatever
privacyStatus we ask for. publish() reports what YouTube actually set, so the page can say so.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

from . import library

DIRNAME = "youtube"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
CATEGORY_SPORTS = "17"
CHUNK = 8 * 1024 * 1024
DEFAULT_DESCRIPTION = ""
PRIVACY = ("private", "unlisted", "public")
DEFAULT_PRIVACY = "unlisted"          # share the link with coaches first; go Public in Studio later

_lock = threading.Lock()
job: dict = {"state": "idle"}           # sign-in or upload, one at a time


def folder() -> Path:
    d = library.home() / DIRNAME
    d.mkdir(exist_ok=True)
    return d


def _state_path() -> Path:
    return folder() / "youtube.json"


def _load() -> dict:
    p = _state_path()
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"description": raw.get("description", DEFAULT_DESCRIPTION), "published": raw.get("published", {}),
            "channel": raw.get("channel"), "privacy": raw.get("privacy") if raw.get("privacy") in PRIVACY else DEFAULT_PRIVACY}


def _save(st: dict) -> None:
    p = _state_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    tmp.replace(p)


def _key(f: Path) -> str:
    s = f.stat()
    return f"{f.resolve()}|{s.st_size}|{int(s.st_mtime)}"


def clean(text: str, limit: int) -> str:
    """YouTube rejects < and > in titles/descriptions; titles max 100 chars, descriptions 5000 bytes."""
    text = re.sub(r"[<>]", "", str(text or "")).strip()
    while len(text.encode("utf-8")) > limit:
        text = text[:-1]
    return text


def default_title(pl: dict | None, vertical: bool, recruiting: bool, when: datetime | None = None) -> str:
    pl = pl or {}
    name = (pl.get("name") or "").strip() or "Highlights"
    num = str(pl.get("number") or "").strip()
    line = (pl.get("tagline") or "").strip() or f"{(when or datetime.now()).year} Highlights"
    t = f"{name} #{num} – {line}" if num else f"{name} – {line}"
    if recruiting:
        t += " – Recruiting"
    if vertical:
        t += " #Shorts"
    return clean(t, 100)


# ---------------- sign-in ----------------
def has_secret() -> bool:
    return (folder() / "client_secret.json").exists()


def store_secret(data: bytes) -> None:
    try:
        raw = json.loads(data)
    except ValueError:
        raise ValueError("that isn't the JSON file Google gives you - download it again from the Credentials page")
    if "installed" not in raw:
        if "web" in raw:
            raise ValueError("that's a 'Web application' client - create one of type 'Desktop app' instead")
        raise ValueError("that isn't an OAuth client file - on the Credentials page use ⤓ Download JSON")
    (folder() / "client_secret.json").write_bytes(data)
    (folder() / "token.json").unlink(missing_ok=True)       # a new client needs a new sign-in


def _save_token(creds) -> None:
    """Write the sign-in atomically: a half-written token.json (app closed mid-write) used to crash
    every YouTube request and hide the Publish box."""
    tok = folder() / "token.json"
    tmp = tok.with_suffix(".json.tmp")
    tmp.write_text(creds.to_json(), encoding="utf-8")
    tmp.replace(tok)


def _creds():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    tok = folder() / "token.json"
    if not tok.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(tok), SCOPES)
    except (ValueError, OSError):        # empty or damaged sign-in: treat as signed out, so Connect shows
        return None
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:  # noqa: BLE001 - revoked or expired (Testing-mode tokens die after 7 days)
            return None
        _save_token(creds)
        return creds
    return None


def _service(creds):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def status() -> dict:
    st = _load()
    out = {"secret": has_secret(), "connected": False, "channel": None, "description": st["description"],
           "privacy": st["privacy"], "job": job}
    creds = _creds() if out["secret"] else None
    if creds:
        out["connected"] = True
        ch = st.get("channel")
        if not ch:
            try:
                items = _service(creds).channels().list(part="snippet", mine=True).execute().get("items") or []
                ch = items[0]["snippet"]["title"] if items else "(no channel on this account)"
                st["channel"] = ch
                _save(st)
            except Exception:  # noqa: BLE001 - still usable, just no name
                ch = None
        out["channel"] = ch
    return out


def connect() -> dict:
    """Open Google's consent page in the browser; finishes in the background."""
    if not has_secret():
        raise ValueError("add the Google client file first")
    with _lock:
        if job.get("state") == "running":
            return job
        job.clear()
        job.update(state="running", kind="connect", step="Waiting for you to sign in to Google in the browser…")

    def run():
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(str(folder() / "client_secret.json"), SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=300,
                                          authorization_prompt_message="",
                                          success_message="Signed in. You can close this tab and go back to Highlight Reels.")
            _save_token(creds)
            _forget_channel()
            job.update(state="done", step="Connected")
        except Exception as e:  # noqa: BLE001 - shown on the page
            job.update(state="error", error=f"Sign-in didn't finish: {e}")
    threading.Thread(target=run, daemon=True).start()
    return job


def _forget_channel() -> None:
    st = _load()
    st["channel"] = None
    _save(st)


def disconnect() -> None:
    (folder() / "token.json").unlink(missing_ok=True)
    _forget_channel()


# ---------------- publishing ----------------
def save_description(text: str) -> str:
    st = _load()
    st["description"] = clean(text, 5000)
    _save(st)
    return st["description"]


def published(f: Path) -> dict | None:
    return _load()["published"].get(_key(f)) if f.exists() else None


def publish(items: list[dict], description: str, privacy: str | None = None) -> dict:
    """items: [{path, title}]. Uploads one after another in the background, each with privacy
    (private / unlisted / public; default: the last one picked, else unlisted)."""
    privacy = privacy or _load()["privacy"]
    if privacy not in PRIVACY:
        raise ValueError("visibility must be private, unlisted or public")
    if job.get("state") == "running":
        raise ValueError("already uploading - wait for it to finish")
    creds = _creds()
    if creds is None:
        raise ValueError("connect your YouTube channel first")
    todo = []
    for it in items:
        f = Path(str(it.get("path") or ""))
        title = clean(it.get("title"), 100)
        if not f.exists():
            raise ValueError(f"file not found: {f.name}")
        if not title:
            raise ValueError(f"give {f.name} a title")
        todo.append((f, title))
    if not todo:
        raise ValueError("tick at least one video")
    description = save_description(description)
    st = _load()
    st["privacy"] = privacy
    _save(st)
    with _lock:
        job.clear()
        job.update(state="running", kind="upload", step="starting", done=0, total=len(todo), frac=0.0, results=[])

    def run():
        from googleapiclient.http import MediaFileUpload
        yt = _service(creds)
        try:
            for i, (f, title) in enumerate(todo):
                job.update(step=f"Uploading {f.name}", done=i, frac=0.0)
                body = {"snippet": {"title": title, "description": description, "categoryId": CATEGORY_SPORTS},
                        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
                req = yt.videos().insert(part="snippet,status", body=body,
                                         media_body=MediaFileUpload(str(f), mimetype="video/mp4",
                                                                    chunksize=CHUNK, resumable=True))
                res = None
                while res is None:
                    prog, res = req.next_chunk(num_retries=3)
                    if prog:
                        job["frac"] = prog.progress()
                vid = res["id"]
                got = (res.get("status") or {}).get("privacyStatus", privacy)
                rec = {"file": f.name, "id": vid, "url": f"https://youtu.be/{vid}",
                       "studio": f"https://studio.youtube.com/video/{vid}/edit", "title": title,
                       "privacy": got, "asked": privacy, "at": datetime.now().isoformat(timespec="minutes")}
                with _lock:
                    st = _load()
                    st["published"][_key(f)] = rec
                    _save(st)
                job["results"].append(rec)
            job.update(state="done", step="done", done=len(todo), frac=1.0)
        except Exception as e:  # noqa: BLE001 - shown on the page
            msg = str(e)
            if "quotaExceeded" in msg or "uploadLimitExceeded" in msg:
                msg = "YouTube's daily upload limit was reached - try the rest tomorrow"
            job.update(state="error", error=msg)
    threading.Thread(target=run, daemon=True).start()
    return job
