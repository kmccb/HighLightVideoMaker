"""Native Windows file/folder dialogs, opened by the local server for the browser page.

A browser can't hand a page the real path of a folder, but the server runs on the same
PC, so it opens the normal dialog itself. One dialog at a time; it's forced on top so it
doesn't open behind the browser.
"""
from __future__ import annotations

import threading
from pathlib import Path

_one = threading.Lock()


def _run(fn) -> str | None:
    import tkinter as tk
    from tkinter import filedialog
    out: list = [None]

    def go():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        try:
            out[0] = fn(filedialog, root) or None
        finally:
            root.destroy()

    with _one:
        t = threading.Thread(target=go, daemon=True)   # Tk wants its own thread here
        t.start()
        t.join()
    return out[0]


def folder(title: str = "Choose the folder with the clips", start: str | None = None) -> Path | None:
    p = _run(lambda fd, root: fd.askdirectory(parent=root, title=title, initialdir=start, mustexist=True))
    return Path(p) if p else None


def music_file(start: str | None = None) -> Path | None:
    types = [("Music", "*.mp3 *.m4a *.wav *.aac *.flac *.ogg"), ("All files", "*.*")]
    p = _run(lambda fd, root: fd.askopenfilename(parent=root, title="Choose a song", initialdir=start, filetypes=types))
    return Path(p) if p else None
