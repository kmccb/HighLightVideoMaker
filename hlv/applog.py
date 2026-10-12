"""The app's activity log: everything the app prints, kept for the Activity panel and on disk.

Two jobs:
- `install()` captures stdout/stderr line by line (memory + `~/HighlightReels/logs/app.log`), so the
  page can show what the app is doing (`/api/log`).
- It also stops the console window from freezing the app. A click inside a Windows console starts
  "select" mode (QuickEdit), and every print then blocks until a key is pressed: a project setup hung
  for good on its first print that way (2026-10-10). QuickEdit is switched off, and console output goes
  through a queue and its own thread, so nothing in the app ever waits for the window.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

KEEP = 3000                 # lines kept in memory
MAX_FILE = 2_000_000        # app.log is rolled to app.old.log past this size

_lock = threading.Lock()
_lines: deque[tuple[int, float, str]] = deque(maxlen=KEEP)
_seq = 0
_file: Path | None = None
_console: queue.Queue | None = None


def add(text: str) -> None:
    """One line into the log (and the log file). Never raises."""
    global _seq
    text = text.rstrip()
    if not text:
        return
    now = time.time()
    with _lock:
        _seq += 1
        _lines.append((_seq, now, text))
        if _file is not None:
            try:
                if _file.exists() and _file.stat().st_size > MAX_FILE:
                    _file.replace(_file.with_name("app.old.log"))
                with open(_file, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + "  " + text + "\n")
            except OSError:
                pass


def say(text: str) -> None:
    """Log a line and show it in the console window too."""
    print(text, flush=True)


def lines(after: int = 0, limit: int = 500) -> dict:
    """Lines numbered above `after` (the newest `limit`), and how long ago the last one was."""
    with _lock:
        out = [{"n": n, "t": t, "text": x} for n, t, x in _lines if n > after][-limit:]
        last_t = _lines[-1][1] if _lines else None
        return {"lines": out, "last": _seq, "quiet": None if last_t is None else round(time.time() - last_t, 1),
                "file": str(_file) if _file else None}


class _Tee:
    """Stands in for sys.stdout / sys.stderr: whole lines go to the log, the text goes on to the console."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""
        self._block = threading.Lock()

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        with self._block:
            self._buf += s
            *whole, self._buf = self._buf.split("\n")
        for ln in whole:
            add(ln)
        if _console is not None and self._stream is not None:
            try:
                _console.put_nowait((self._stream, s))
            except queue.Full:
                pass          # the window isn't taking output; the log has it
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _console_writer():
    while True:
        stream, s = _console.get()
        try:
            stream.write(s)
            stream.flush()
        except Exception:  # noqa: BLE001 - a closed or broken console must not matter
            pass


def _no_quick_edit() -> None:
    """Windows: a click in the console window must not pause the app."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)                     # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            # clear ENABLE_QUICK_EDIT_MODE (0x40); ENABLE_EXTENDED_FLAGS (0x80) makes that take effect
            k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except Exception:  # noqa: BLE001
        pass


def install(home: Path) -> None:
    """Start logging to `home/logs/app.log` and capture everything printed from here on."""
    global _file, _console
    if _console is not None:
        return
    logs = Path(home) / "logs"
    try:
        logs.mkdir(parents=True, exist_ok=True)
        _file = logs / "app.log"
    except OSError:
        _file = None
    _no_quick_edit()
    _console = queue.Queue(maxsize=5000)
    threading.Thread(target=_console_writer, daemon=True).start()
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    add("---- app started ----")
