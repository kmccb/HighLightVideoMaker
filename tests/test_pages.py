"""The pages' scripts must at least parse: one stray apostrophe in a string broke the whole
review page once (2026-10-04)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "hlv" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to parse the scripts")
@pytest.mark.parametrize("page", sorted(p.name for p in STATIC.glob("*.html")))
def test_page_scripts_parse(page, tmp_path):
    html = (STATIC / page).read_text(encoding="utf-8")
    for i, js in enumerate(re.findall(r"<script>([\s\S]*?)</script>", html)):
        f = tmp_path / f"{page}-{i}.mjs"          # module: top-level await is allowed
        f.write_text(js, encoding="utf-8")
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{page} script {i}: {r.stderr.strip()[:400]}"
