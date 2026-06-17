# WisprFlow clone with Hammerspoon + `assembly dictate`

A ~100-line [Hammerspoon](https://www.hammerspoon.org) script that turns the
AssemblyAI CLI into a [WisprFlow](https://wisprflow.ai)-style dictation tool:
**hold a hotkey, speak, release, and the transcript is typed in at your
cursor** — in any app, no terminal in sight.

It works because `assembly dictate` is built for exactly this. It starts
recording immediately and prints the transcript to stdout when it receives
`SIGTERM`, so a hotkey tool can drive it as a background task:

```text
hotkey down  ->  launch `assembly dictate` (records the mic)
hotkey up    ->  task:terminate()  (SIGTERM)  ->  dictate prints the transcript
             ->  Hammerspoon reads stdout and pastes it at the cursor
```

## Setup

1. **Install the AssemblyAI CLI** and sign in (the transcript needs an API key):

   ```sh
   brew tap assemblyai/cli https://github.com/AssemblyAI/cli
   brew install assembly
   assembly login            # or export ASSEMBLYAI_API_KEY=...
   ```

   Confirm dictation works from the terminal first — say a few words and press
   `Ctrl-C`'s gentler sibling, `kill -TERM`, from another shell, or just:

   ```sh
   assembly dictate
   # speak, then in another terminal: kill -TERM $(pgrep -f 'assembly dictate')
   ```

2. **Install Hammerspoon** and grant it Accessibility permission (System
   Settings → Privacy & Security → Accessibility) so it can send keystrokes:

   ```sh
   brew install --cask hammerspoon
   ```

3. **Drop the script in place.** Copy `wisprflow.lua` to `~/.hammerspoon/` and
   load it from your `~/.hammerspoon/init.lua`:

   ```lua
   require("wisprflow")
   ```

   Then reload the config (Hammerspoon menubar → *Reload Config*).

## Use it

Hold **⌃⌥D** (control + option + D), speak, and release. A 🔴 appears in the
menubar while recording; on release the text lands at your cursor. macOS will
prompt once for microphone access.

## Customize

Edit the config block at the top of `wisprflow.lua`:

| Setting | What it does |
| --- | --- |
| `M.hotkey` | The push-to-talk chord (default `⌃⌥D`). Hammerspoon can't bind a bare `fn` key, so use a modifier+letter. |
| `M.dictateArgs` | Extra `assembly dictate` flags, e.g. `{ "--language", "es" }` to dictate in Spanish, or repeated `--word-boost` to bias tricky terms. |
| `M.insertMode` | `"paste"` (clipboard + ⌘V, fast, restores your clipboard) or `"type"` (simulated keystrokes, works everywhere). |
| `M.sounds` | Play the mic-open / done chimes for audible feedback. |

## Notes

- **PATH**: Hammerspoon launches with a minimal `PATH`, so the script resolves
  `assembly` through a login shell (`command -v assembly`). If you installed the
  CLI somewhere unusual, hard-code the path in `resolveAssembly()`.
- **Limits**: each utterance is capped at 120 s (the Sync STT API limit); pass
  `--max-seconds` to stop sooner.
- **Toggle instead of hold**: prefer tap-to-start / tap-to-stop? Bind only the
  pressed callback to a function that calls `startDictation()` when idle and
  `stopDictation()` when a recording is in flight.
- **Pipe it somewhere else**: the same SIGTERM trick powers shell pipelines —
  `assembly dictate | assembly llm "write a conventional commit"`. See
  `assembly dictate --help` for more.
