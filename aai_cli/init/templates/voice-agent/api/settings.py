from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# `aai init` writes this for you; defaults to production. Host only, no scheme.
AGENTS_HOST = os.environ.get("ASSEMBLYAI_AGENTS_HOST", "agents.assemblyai.com")
TOKEN_EXPIRES_IN_SECONDS = 60
TOKEN_PATH = "/v1/token"
WEBSOCKET_PATH = "/v1/ws"
