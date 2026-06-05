"""Named help panels for `aai --help`.

Rich groups top-level commands under these headings (via each command's
``rich_help_panel``), so the root help reads as a journey rather than a flat
list — the same approach the Vercel and Supabase CLIs take. Panels render in
the order their first command appears (see ``_COMMAND_ORDER`` in ``main.py``);
most-used commands first, account/setup last.

Centralized here so the heading strings have one source of truth — a typo in a
decorator would otherwise silently spawn a duplicate panel.
"""

from __future__ import annotations

QUICK_START = "Quick Start"  # zero-to-running onboarding: init
TRANSCRIPTION = "Transcription & AI"  # the verbs you run: transcribe, stream, agent, llm
HISTORY = "History"  # browse past work: transcripts, sessions
ACCOUNT = "Account"  # auth, billing, keys: login/logout/whoami, balance/usage/limits, keys, audit
SETUP = "Setup & Tools"  # get set up & maintain: samples, doctor, claude, version
