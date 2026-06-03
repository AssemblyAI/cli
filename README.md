# AssemblyAI CLI (`aai`)

Onboarding CLI for AssemblyAI: `aai login` then `aai transcribe --sample`.

## Install (dev)

    pip install -e ".[dev]"

## Usage

    aai login
    aai transcribe --sample

## Streaming

Real-time transcription from a file (no extra dependency):

    aai stream path/to/audio.wav        # 16 kHz mono WAV streams directly
    aai stream path/to/audio.mp3        # other formats require ffmpeg on PATH

From the microphone (install the optional extra first):

    pip install "assemblyai-cli[mic]"
    aai stream                          # Ctrl-C to stop

Add `--json` for newline-delimited JSON events (also the default when piped or run by an agent).

## AI coding agents

Wire Claude Code up to AssemblyAI's live docs (MCP server) and the AssemblyAI
skill so your agent writes current, correct integration code:

    aai claude install           # installs the docs MCP server + skill (user scope)
    aai claude status            # show what's wired up
    aai claude remove            # unwind both

`install` shells out to `claude mcp add` for the docs MCP server and to
`npx skills add` for the skill. Pass `--scope project` to scope the MCP server
to the current project instead of the whole machine. A missing `claude` or
`npx` is reported and skipped (with the manual command to run), not treated as
an error.

## Voice agent

Have a live, two-way voice conversation with an AssemblyAI voice agent (requires the
`[mic]` extra for microphone + speaker audio):

    pip install "assemblyai-cli[mic]"
    aai agent                              # talk; the agent talks back. Ctrl-C to stop.
    aai agent --voice james --greeting "Hi there"
    aai agent --prompt-file persona.txt    # load the system prompt from a file
    aai agent --list-voices                # see available voices

By default the agent runs **half-duplex**: your mic mutes while the agent is speaking,
so it can't hear itself on your speakers. With headphones, add `--full-duplex` for
true barge-in (interrupt the agent mid-sentence). Add `--json` for newline-delimited
JSON events.
