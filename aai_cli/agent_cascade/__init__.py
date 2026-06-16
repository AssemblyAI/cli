"""The terminal *agent cascade* slice: a client-orchestrated voice cascade.

`assembly agent-cascade` holds the same kind of live voice conversation as
`assembly agent`, but where `agent` talks to AssemblyAI's single Voice Agent
endpoint, this slice wires the three primitives together itself — Streaming STT
-> the LLM Gateway -> streaming TTS — exactly like the ``agent-cascade``
``assembly init`` template does server-side. Because it uses streaming TTS it is
sandbox-only.

`engine.run_cascade` is the orchestrator; it takes injected dependencies
(`CascadeDeps`) so tests drive the whole cascade against fakes, the same seam
`aai_cli/tts/session.py` uses.
"""

from __future__ import annotations
