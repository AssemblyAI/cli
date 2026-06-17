# Meeting notes with Hammerspoon + `assembly stream`

Record a meeting with **one hotkey** and get a live, auto-named Markdown note
out the other end — system audio *and* your mic, diarized by speaker, summarized
by an LLM as you talk. This is the [Hammerspoon](https://www.hammerspoon.org)
front-end to the [`justfile`](justfile) in this directory, so you can drive the
same workflow from a global hotkey or from the terminal.

It leans on one command:

```sh
assembly stream --system-audio --speaker-labels --auto-name \
  --save-dir ~/meeting-notes --model claude-sonnet-4-6 \
  --llm "Keep running meeting notes: summary, decisions, action items, open questions"
```

`--system-audio` mixes the meeting app's output with your mic as separate
diarized speakers; `--auto-name` names the file from its content and buckets it
under `<save-dir>/YYYY-MM-DD/`; `--llm` re-runs the prompt over the growing
transcript so the `.md` note updates live. The note is written on a clean stop —
Ctrl-C from the terminal, or `SIGTERM` from a hotkey tool (`stream` routes both
to the same save path).

## The justfile

If you live in the terminal, the `justfile` gives you three recipes (needs
[`just`](https://github.com/casey/just); `list` also wants `fzf` + `glow`):

```sh
just record                                  # record a meeting + its note
just list                                    # browse notes, preview rendered
just search "what did we decide on pricing?" # ask across all notes (LLM cites them)
```

## The Hammerspoon hotkey

For a global, no-terminal version, use `meeting_notes.lua`:

1. **Install the CLI** and sign in, then **install Hammerspoon** and grant it
   Accessibility + Screen Recording permission (System Settings → Privacy &
   Security) — system-audio capture needs Screen Recording:

   ```sh
   brew tap assemblyai/cli https://github.com/AssemblyAI/cli
   brew install assembly
   assembly login
   brew install --cask hammerspoon
   ```

2. **Drop the script in place.** Copy `meeting_notes.lua` to `~/.hammerspoon/`
   and load it from your `~/.hammerspoon/init.lua`:

   ```lua
   require("meeting_notes")
   ```

   Then reload the config (Hammerspoon menubar → *Reload Config*).

3. **Use it.** Press **⌃⌥M** to start recording (a `🔴 REC` badge appears in the
   menubar); press it again to stop and save the note. **⌃⌥⇧M** opens the notes
   folder in Finder.

## Customize

Edit the config block at the top of `meeting_notes.lua` (it mirrors the
justfile's variables):

| Setting | What it does |
| --- | --- |
| `M.hotkey` | The toggle-recording chord (default `⌃⌥M`). |
| `M.openHotkey` | The "open notes folder" chord (default `⌃⌥⇧M`). |
| `M.transcriptsDir` | Where notes are saved (default `~/meeting-notes`). |
| `M.model` | The model used for naming and the live note. |
| `M.notePrompt` | The `--llm` instruction that shapes each note. |
| `M.sounds` | Play the start/stop chimes for audible feedback. |

## Notes

- **PATH**: Hammerspoon launches with a minimal `PATH`, so the script resolves
  `assembly` through a login shell. Hard-code the path in `resolveAssembly()` if
  the CLI lives somewhere unusual.
- **Permissions**: capturing system audio requires the Screen Recording
  permission in addition to Accessibility; macOS prompts on first use.
- **Searching later**: the justfile's `search` recipe points `assembly llm` at
  the notes directory — it recurses for `.md`/`.txt` files and answers with
  citations, so your meeting history becomes queryable.
