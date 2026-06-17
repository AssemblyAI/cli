-- One-hotkey meeting notes on macOS, powered by Hammerspoon
-- (https://www.hammerspoon.org) and `assembly stream`.
--
-- This is the Hammerspoon companion to the justfile in this directory: a hotkey
-- toggles a recording that captures system audio + your mic with speaker labels,
-- auto-names the file from its content, and writes a live Markdown note via an
-- LLM. `assembly stream` saves the note on a clean stop, and SIGTERM (which
-- `hs.task:terminate()` sends) is routed to that same stop path, so:
--
--   hotkey  ->  start `assembly stream …` recording the meeting (note builds live)
--   hotkey  ->  task:terminate() (SIGTERM) -> stream flushes + saves the .md note
--
-- A second hotkey opens the notes folder in Finder. Load it from your
-- ~/.hammerspoon/init.lua with:
--
--   require("meeting_notes")        -- if this file is at ~/.hammerspoon/meeting_notes.lua
--
-- then reload the Hammerspoon config (menubar -> Reload Config).

local M = {}

-- ---------------------------------------------------------------------------
-- Configuration — tweak these to taste. The defaults mirror the justfile.
-- ---------------------------------------------------------------------------

-- Toggle-recording hotkey. Default: ⌃⌥M (control+option+M). Tap to start, tap
-- again to stop and save the note.
M.hotkey = { mods = { "ctrl", "alt" }, key = "m" }

-- "Open notes folder" hotkey. Default: ⌃⌥⇧M.
M.openHotkey = { mods = { "ctrl", "alt", "shift" }, key = "m" }

-- Where notes land. The CLI buckets each note under <dir>/YYYY-MM-DD/ and names
-- the file from the meeting's content (--auto-name). "~" is expanded for you.
M.transcriptsDir = "~/meeting-notes"

-- Model used both for naming and for the live note.
M.model = "claude-sonnet-4-6"

-- The live-note instruction handed to --llm. `assembly stream` re-runs it over
-- the growing transcript, so the note updates in place as the meeting goes.
M.notePrompt = "Keep running meeting notes: a one-line summary, decisions, "
  .. "action items (with owners), and open questions. Update as we talk."

-- Play the system start/stop chimes for audible feedback.
M.sounds = true

-- ---------------------------------------------------------------------------
-- Internals.
-- ---------------------------------------------------------------------------

-- Resolve the `assembly` binary through a login shell so Homebrew's PATH (e.g.
-- /opt/homebrew/bin) is on it — Hammerspoon itself starts with a minimal PATH.
local function resolveAssembly()
  local path = hs.execute("command -v assembly", true)
  path = path and path:gsub("%s+$", "") or ""
  if path == "" then
    hs.notify
      .new({
        title = "Meeting notes (Hammerspoon)",
        informativeText = "`assembly` not found on PATH. Install the AssemblyAI CLI first.",
      })
      :send()
    return nil
  end
  return path
end

-- Expand a leading "~" to the user's home directory.
local function expanduser(p)
  if p:sub(1, 1) == "~" then
    return os.getenv("HOME") .. p:sub(2)
  end
  return p
end

local assemblyPath = resolveAssembly()
local notesDir = expanduser(M.transcriptsDir)
local task = nil -- the in-flight `assembly stream` hs.task, or nil
local indicator = nil -- a menubar dot shown while recording

local function notify(text)
  hs.notify.new({ title = "Meeting notes (Hammerspoon)", informativeText = text }):send()
end

local function showRecording()
  if not indicator then
    indicator = hs.menubar.new()
  end
  if indicator then
    indicator:setTitle("🔴 REC")
    indicator:setTooltip("Recording the meeting… press the hotkey again to save the note")
  end
  if M.sounds then
    hs.sound.getByName("Tink"):play()
  end
end

local function hideRecording()
  if indicator then
    indicator:delete()
    indicator = nil
  end
  if M.sounds then
    hs.sound.getByName("Pop"):play()
  end
end

local function startRecording()
  if not assemblyPath then
    return
  end
  local args = {
    "stream",
    "--system-audio",
    "--speaker-labels",
    "--auto-name",
    "--save-dir",
    notesDir,
    "--model",
    M.model,
    "--llm",
    M.notePrompt,
  }
  task = hs.task.new(assemblyPath, function(_, _, stdErr)
    -- stream exits 130 on a clean (SIGTERM/Ctrl-C) stop, with the note saved.
    local wasRecording = indicator ~= nil
    task = nil
    hideRecording()
    if wasRecording then
      notify("Saved a meeting note to " .. M.transcriptsDir)
    elseif stdErr and stdErr ~= "" then
      notify(stdErr)
    end
  end, args)
  if task:start() then
    showRecording()
  else
    task = nil
    notify("Couldn't start `assembly stream`.")
  end
end

local function stopRecording()
  if task and task:isRunning() then
    task:terminate() -- SIGTERM -> stream's clean-stop path -> the note is saved
  end
end

-- One hotkey toggles recording: start when idle, stop+save when in flight.
hs.hotkey.bind(M.hotkey.mods, M.hotkey.key, function()
  if task and task:isRunning() then
    stopRecording()
  else
    startRecording()
  end
end)

-- A second hotkey reveals the notes folder in Finder.
hs.hotkey.bind(M.openHotkey.mods, M.openHotkey.key, function()
  hs.execute("mkdir -p '" .. notesDir .. "' && open '" .. notesDir .. "'", true)
end)

return M
