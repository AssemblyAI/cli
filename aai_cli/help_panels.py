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

QUICK_START = "Quick Start"  # zero-to-running onboarding: onboard
BUILD = "Build an App"  # scaffold a new project: init
TRANSCRIPTION = "Run AssemblyAI"  # use AssemblyAI directly: transcribe, stream, agent, llm
HISTORY = "History"  # browse past work: transcripts, sessions
ACCOUNT = "Account"  # auth, billing, keys: login/logout/whoami, balance/usage/limits, keys, audit
SETUP = "Setup & Tools"  # get set up & maintain: doctor, setup

# Option panels group a single command's flags within its own ``--help``. The
# `transcribe` command exposes 40+ options; without panels they render as one
# flat wall. Each ``typer.Option(rich_help_panel=...)`` files the flag under one
# of these headings; flags left unpanelled fall in Rich's default "Options"
# panel — we keep the everyday ones (source, --sample, --json, -o, --show-code)
# there so the common case stays at the top.
OPT_MODEL = "Model & Language"
OPT_FORMATTING = "Formatting"
OPT_SPEAKERS = "Speakers & Channels"
OPT_GUARDRAILS = "Guardrails"
OPT_ANALYSIS = "Analysis"
OPT_CUSTOMIZATION = "Customization"
OPT_WEBHOOKS = "Webhooks"
OPT_TRANSLATION = "Translation"
OPT_ADVANCED = "Advanced"
OPT_LLM = "LLM Transform"
# stream-specific panels (real-time concerns that file transcription has no equivalent for)
OPT_CAPTURE = "Audio Capture"
OPT_TURNS = "Turn Detection"
OPT_FEATURES = "Features"
