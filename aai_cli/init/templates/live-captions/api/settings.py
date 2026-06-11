from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# `assembly init` writes this for you; defaults to production. Host only, no scheme.
STREAMING_HOST = os.environ.get("ASSEMBLYAI_STREAMING_HOST", "streaming.assemblyai.com")
TOKEN_EXPIRES_IN_SECONDS = 60
TOKEN_PATH = "/v3/token"
WEBSOCKET_PATH = "/v3/ws"
