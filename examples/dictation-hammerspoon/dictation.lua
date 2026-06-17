-- Push-to-talk dictation anywhere on macOS (a WisprFlow-style clone), powered
-- by Hammerspoon (https://www.hammerspoon.org) and `assembly dictate`.
--
-- Hold the hotkey to record, release to transcribe. The text is inserted at the
-- cursor in whatever app has focus — a chat box, your editor, a search field.
-- `assembly dictate` records immediately and prints the transcript when it
-- receives SIGTERM (which `hs.task:terminate()` sends), so the whole flow is:
--
--   hotkey down  -> launch `assembly dictate` as a background task
--   hotkey up    -> terminate it (SIGTERM) -> read stdout -> type it at the cursor
--
-- Load it from your ~/.hammerspoon/init.lua with:
--
--   require("dictation")            -- if this file is at ~/.hammerspoon/dictation.lua
--
-- then reload the Hammerspoon config (menubar -> Reload Config).

local M = {}

-- ---------------------------------------------------------------------------
-- Configuration — tweak these to taste.
-- ---------------------------------------------------------------------------

-- Push-to-talk hotkey. Default: hold ⌃⌥ (control+option) + D.
-- Hammerspoon can't bind a bare modifier like WisprFlow's fn key, so we use a
-- modifier+letter chord; pick anything that doesn't clash with your apps.
M.hotkey = { mods = { "ctrl", "alt" }, key = "d" }

-- Extra arguments passed to every `assembly dictate` run, e.g.:
--   { "--language", "es" }                       -- dictate in Spanish
--   { "--word-boost", "AssemblyAI", "--word-boost", "LeMUR" }  -- bias terms
M.dictateArgs = {}

-- How to insert the transcript:
--   "type"  — simulate keystrokes (works everywhere, slower for long text)
--   "paste" — stash it on the clipboard and ⌘V (fast, restores your clipboard)
M.insertMode = "paste"

-- Play the system mic-open / done sounds for audible feedback.
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
        title = "Dictation (Hammerspoon)",
        informativeText = "`assembly` not found on PATH. Install the AssemblyAI CLI first.",
      })
      :send()
    return nil
  end
  return path
end

local assemblyPath = resolveAssembly()
local task = nil -- the in-flight `assembly dictate` hs.task, or nil
local indicator = nil -- a menubar dot shown while recording

local function showRecording()
  if not indicator then
    indicator = hs.menubar.new()
  end
  if indicator then
    indicator:setTitle("🔴")
    indicator:setTooltip("Dictating… release the hotkey to transcribe")
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
end

local function insert(text)
  text = text:gsub("%s+$", "") -- drop the trailing newline `dictate` prints
  if text == "" then
    return
  end
  if M.insertMode == "paste" then
    local saved = hs.pasteboard.getContents()
    hs.pasteboard.setContents(text)
    hs.eventtap.keyStroke({ "cmd" }, "v")
    -- Restore the previous clipboard after the paste lands.
    hs.timer.doAfter(0.2, function()
      hs.pasteboard.setContents(saved)
    end)
  else
    hs.eventtap.keyStrokes(text)
  end
end

-- Hotkey pressed: start recording (no-op if a run is already in flight).
local function startDictation()
  if task or not assemblyPath then
    return
  end
  local args = { "dictate" }
  for _, a in ipairs(M.dictateArgs) do
    args[#args + 1] = a
  end
  task = hs.task.new(assemblyPath, function(_, stdOut, stdErr)
    task = nil
    hideRecording()
    if M.sounds then
      hs.sound.getByName("Pop"):play()
    end
    if stdOut and stdOut ~= "" then
      insert(stdOut)
    elseif stdErr and stdErr ~= "" then
      hs.notify.new({ title = "Dictation (Hammerspoon)", informativeText = stdErr }):send()
    end
  end, args)
  if task:start() then
    showRecording()
  else
    task = nil
  end
end

-- Hotkey released: SIGTERM the recording so `dictate` transcribes and exits.
local function stopDictation()
  if task and task:isRunning() then
    task:terminate() -- sends SIGTERM, which `assembly dictate` treats as "done"
  end
end

hs.hotkey.bind(M.hotkey.mods, M.hotkey.key, startDictation, stopDictation)

return M
