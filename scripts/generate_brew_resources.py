"""Emit Homebrew resource stanzas for aai-cli's runtime closure from uv.lock.
FALLBACK ONLY: does not evaluate platform markers. See plan Task 3 Step 3."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text())
pkgs = {p["name"]: p for p in lock["package"]}

root = pkgs["aai-cli"]
queue = [d["name"] for d in root.get("dependencies", [])]
seen: set[str] = set()
while queue:
    name = queue.pop()
    if name in seen:
        continue
    seen.add(name)
    queue.extend(d["name"] for d in pkgs.get(name, {}).get("dependencies", []))
seen.discard("aai-cli")

# Windows-only / drop; Linux-only / caller must wrap in on_linux:
WINDOWS_ONLY = {"pywin32-ctypes"}

out: list[str] = []
for name in sorted(seen):
    if name in WINDOWS_ONLY:
        continue
    sdist = pkgs[name].get("sdist")
    if not sdist:
        out.append(f"# WARNING: {name} has no sdist (wheel-only) — handle manually\n")
        continue
    digest = sdist["hash"].removeprefix("sha256:")
    out.append(
        f'  resource "{name}" do\n    url "{sdist["url"]}"\n    sha256 "{digest}"\n  end\n\n'
    )

sys.stdout.write("".join(out))
