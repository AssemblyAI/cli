"""Detached-subprocess plumbing shared by the background flushers.

Both telemetry delivery (`assembly telemetry flush`) and the update-check refresh
(`assembly _update-check`) run as detached children so the user's command never
waits on the network. The spawn recipe lives here so the two can't drift on the
parts that make it safe: own session, discarded stdio, and a self-disable env var
so a child can never spawn another of itself.
"""

from __future__ import annotations

import os
import subprocess
import sys


def spawn_detached(cli_args: list[str], *, disable_env_var: str) -> None:
    """Spawn ``python -m aai_cli <cli_args>`` detached; return immediately.

    ``disable_env_var`` is set to ``"1"`` in the child's environment to suppress
    the spawning subsystem there. S603 is ignored project-wide for the CLI's own
    shell-outs.
    """
    subprocess.Popen(
        [sys.executable, "-m", "aai_cli", *cli_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, disable_env_var: "1"},
    )
