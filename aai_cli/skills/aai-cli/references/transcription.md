# Transcription & AI

Five commands. All accept `--json` (auto-enabled when piped); `transcribe`,
`stream`, `agent`, and `llm` accept `-o/--output` to print a single field.
`transcribe`, `stream`, and `agent` accept `--show-code` to print equivalent
Python SDK code without calling the API.

## `assembly transcribe [SOURCE]` — file / URL / YouTube / podcast page

`SOURCE` is a local file path, public URL, or a media-page URL yt-dlp can extract
(YouTube, Apple Podcasts, Spreaker, SoundCloud, …) — those are downloaded first.
Use `--sample` for the hosted `wildfires.mp3`. Analysis results (summary,
chapters, sentiment, …) render automatically in human mode.

High-value flags (run `assembly transcribe --help` for the full set):

- Model/language: `--speech-model` (best, nano, slam-1, universal),
  `--language-code en_us`, `--language-detection`.
- Diarization: `--speaker-labels`, `--speakers-expected N`, `--multichannel`.
- PII: `--redact-pii`, `--redact-pii-policy person_name,...`,
  `--redact-pii-sub hash|entity_name`, `--redact-pii-audio`.
- Audio intelligence: `--summarization`, `--auto-chapters`,
  `--sentiment-analysis`, `--entity-detection`, `--auto-highlights`,
  `--topic-detection`, `--content-safety`.
- Escape hatch to any SDK field: `--config KEY=VALUE` (repeatable) and
  `--config-file config.json`.
- Post-process: `--llm "PROMPT"` (repeatable; chains over the transcript via LLM
  Gateway), `--translate-to es` (repeatable).
- Output: `-o text|id|status|utterances|srt|vtt|json`, `--chars-per-caption N`
  (caption line length, with `-o srt`/`-o vtt`), `--json`, `--show-code`.

Examples:

```bash
assembly transcribe call.mp3
assembly transcribe --sample
assembly transcribe call.mp3 --speaker-labels --speakers-expected 2 --redact-pii
assembly transcribe call.mp3 -o text
assembly transcribe call.mp3 --show-code
```

## `assembly stream [SOURCE]` — live real-time transcription

Omit `SOURCE` to use the microphone; pass a file/URL/media page to stream that, or
`--sample`. macOS can capture system audio with `--system-audio` (mic + system)
or `--system-audio-only`.

High-value flags (run `assembly stream --help` for the full set):

- Capture: `--device N`, `--sample-rate HZ`, `--encoding pcm_s16le|pcm_mulaw`.
- Model/turns: `--speech-model` (default `u3-rt-pro`), `--format-turns`,
  `--include-partial-turns`, `--end-of-turn-confidence`, `--min-turn-silence`,
  `--max-turn-silence`, `--vad-threshold`.
- Features: `--speaker-labels`, `--max-speakers`, `--keyterms-prompt`,
  `--redact-pii`, `--voice-focus near_field|far_field`, `--domain medical`.
- Live LLM: `--llm "PROMPT"` (refreshes the answer on every finalized turn).
- Output: `-o text|json`, `--json` (newline-delimited JSON events),
  `--show-code`.

Examples:

```bash
assembly stream
assembly stream --system-audio
assembly stream --sample
assembly stream --llm "summarize action items"
assembly stream -o text                 # finalized turns as plain lines, pipe-friendly
```

## `assembly agent [SOURCE]` — full-duplex voice agent

Two-way voice conversation (mic in, TTS out). Pass a file/URL or `--sample` to
speak a recorded clip instead of the mic; the session then ends after the reply.

> **`assembly agent` only *runs* a live conversation in the terminal — it does not
> create any code or project.** If the goal is to *build* a voice-agent app,
> use `assembly init` with the `voice-agent` template (see `setup.md`), not this
> command.

High-value flags:

- `--voice ivy` (see `--list-voices`), `--system-prompt "..."` or
  `--system-prompt-file path`, `--greeting "..."`, `--device N`.
- Output: `-o text|json`, `--json`, `--show-code`.

Examples:

```bash
assembly agent
assembly agent --voice james --greeting "Hi there"
assembly agent --list-voices
assembly agent --show-code
```

## `assembly llm [PROMPT]` — LLM Gateway

Send a prompt to the LLM Gateway. With `--transcript-id ID` the transcript's
text is injected server-side so you can ask questions about a past
transcription. Reads stdin when piped.

High-value flags:

- `--model` (default `claude-haiku-4-5-20251001`, see `--list-models`),
  `--transcript-id ID`, `--system "..."`, `--max-tokens N`.
- `-f/--follow`: re-run the prompt over a transcript growing on stdin,
  refreshing the answer in place on every finalized turn.
- Output: `-o text|json`, `--json`.

Examples:

```bash
assembly llm "summarize" --transcript-id 5551234-abcd
echo "meeting notes" | assembly llm "turn into action items"
assembly stream -o text | assembly llm -f "summarize action items as I talk"
assembly llm --list-models
```

## `assembly clip MEDIA` — cut a media file by transcript content

Cuts clips out of an audio/video file with ffmpeg (must be installed). `MEDIA`
is a local file or a YouTube/media-page URL (audio downloaded via yt-dlp; the
clips then land in `--out-dir` or the current directory). `--speaker`/`--search`
select diarized utterances — the file is transcribed with speaker labels on the
fly, or pass `-t/--transcript-id` (an id, or `-` to read an id or
`transcribe --json` output from stdin). `--llm "instruction"` sends the
timestamped utterances to LLM Gateway and the model picks the windows.
`--range START-END` adds explicit windows (seconds or `[HH:]MM:SS`).
Overlapping selections merge, and clip boundaries snap into nearby silence
(one ffmpeg `silencedetect` pass) so cuts don't land mid-word; each surviving
segment is written as `<name>.clipNN<ext>`.

High-value flags:

- Selection: `--speaker A` (repeatable), `--search "topic"` (case-insensitive),
  `--llm "the best moments"` (composes with the filters), `--range 1:30-2:45`
  (repeatable).
- LLM: `--model` (default `claude-haiku-4-5-20251001`), `--max-tokens N`.
- Shaping: `--padding 0.5` (seconds around each clip), `--no-snap` (cut at the
  exact selected times instead of snapping into silence), `--out-dir clips/`.
- Output: `--json` (paths + start/end/duration of each clip written).

Examples:

```bash
assembly clip meeting.mp4 --speaker A
assembly clip call.mp3 --search "pricing" --padding 0.5
assembly clip talk.mp4 --range 1:30-2:45 --range 10:00-10:30
assembly clip "https://youtube.com/watch?v=ID" --llm "the strongest quote"
assembly transcribe meeting.mp4 --speaker-labels --json | assembly clip meeting.mp4 -t - --llm "the funniest exchange"
```
