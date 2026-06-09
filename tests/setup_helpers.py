"""Shared scaffolding for the `aai setup` test modules.

Not a test module itself (no ``test_`` prefix, so pytest won't collect it): it
holds the fakes and helpers reused across ``test_setup.py`` and
``test_setup_install.py`` so neither file has to redeclare them.
"""

import json
import shutil
import subprocess
from pathlib import Path


def _skill_path() -> Path:
    return Path.home() / ".claude" / "skills" / "assemblyai"


def _cli_skill_path() -> Path:
    return Path.home() / ".claude" / "skills" / "aai-cli"


class FakeRun:
    """Records subprocess calls and returns canned CompletedProcess results.

    `returncodes` maps a command prefix tuple (the first N argv tokens) to a
    return code; the longest matching prefix wins, default 0. To mimic the real
    `skills` CLI, a successful `npx … add` materializes the assemblyai skill under
    HOME (so `install_skill`'s filesystem check passes) and `npx … remove`
    deletes it — toggle with `creates_skill` / `removes_skill`. The aai-cli skill
    is bundled and copied directly (no subprocess), so it never goes through here.
    """

    def __init__(self, returncodes=None, *, creates_skill=True, removes_skill=True):
        self.calls = []
        self.returncodes = returncodes or {}
        self.creates_skill = creates_skill
        self.removes_skill = removes_skill

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        rc = 0
        best = -1
        for prefix, code in self.returncodes.items():
            n = len(prefix)
            if tuple(cmd[:n]) == prefix and n > best:
                rc, best = code, n
        if rc == 0 and cmd[:1] == ["npx"]:
            if "add" in cmd and self.creates_skill:
                _skill_path().mkdir(parents=True, exist_ok=True)
                (_skill_path() / "SKILL.md").write_text("# AssemblyAI")
            elif "remove" in cmd and self.removes_skill:
                shutil.rmtree(_skill_path(), ignore_errors=True)
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="boom")


def _all_tools_present(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: f"/usr/bin/{tool}",
    )


def _statuses(result):
    return {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
