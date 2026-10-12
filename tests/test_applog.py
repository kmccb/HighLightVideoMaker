import queue
import threading

from hlv import applog


def test_lines_are_numbered_and_filtered(tmp_path, monkeypatch):
    monkeypatch.setattr(applog, "_file", tmp_path / "app.log")
    start = applog.lines()["last"]
    applog.add("first")
    applog.add("   ")            # blank lines aren't kept
    applog.add("second\n")
    got = applog.lines(after=start)
    assert [l["text"] for l in got["lines"]] == ["first", "second"]
    assert got["last"] == start + 2
    assert applog.lines(after=start + 1)["lines"][0]["text"] == "second"
    saved = (tmp_path / "app.log").read_text(encoding="utf-8").splitlines()
    assert [s.split("  ", 1)[1] for s in saved] == ["first", "second"]


def test_print_never_waits_for_a_frozen_console(monkeypatch):
    """A console in select mode blocks writes for good; the app's prints must still return."""
    class Frozen:
        def write(self, s):
            threading.Event().wait()     # never returns

        def flush(self):
            pass

    monkeypatch.setattr(applog, "_file", None)
    monkeypatch.setattr(applog, "_console", queue.Queue(maxsize=3))
    threading.Thread(target=applog._console_writer, daemon=True).start()
    tee = applog._Tee(Frozen())
    start = applog.lines()["last"]
    done = threading.Event()

    def chatter():
        for i in range(50):
            tee.write(f"line {i}\n")
        tee.write("half a ")
        tee.write("line\n")
        done.set()
    threading.Thread(target=chatter, daemon=True).start()
    assert done.wait(5), "print blocked on the console"
    texts = [l["text"] for l in applog.lines(after=start)["lines"]]
    assert texts[0] == "line 0" and texts[-1] == "half a line" and len(texts) == 51
