const SESSION_CONFIG = {
  systemPrompt: "You are a friendly, concise voice assistant. Keep replies short and conversational.",
  greeting: "Hi! I'm your AssemblyAI voice agent. What can I help you with?",
  input: { format: { encoding: "audio/pcm" } },
  output: { voice: "ivy", format: { encoding: "audio/pcm" } },
  inputSampleRate: 24000,
  outputSampleRate: 24000,
  processorBufferSize: 4096,
  microphone: { audio: { echoCancellation: true, noiseSuppression: false } },
};

const connBtn = document.getElementById("conn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws = null;
let micPipeline = null;
let player = null;
let connected = false;

connBtn.addEventListener("click", () => (connected ? hangup() : connect().catch(fail)));

function setStatus(message, state) {
  statusEl.textContent = message;
  statusEl.className = state;
}

async function connect() {
  setStatus("Connecting...", "idle");
  const res = await fetch("/api/token", { method: "POST" });
  if (!res.ok) return fail(await res.text());
  const { token, ws_url } = await res.json();

  ws = new WebSocket(`${ws_url}?token=${encodeURIComponent(token)}`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "session.update", session: buildSessionConfig() }));
    startMic().catch(fail);
  };
  ws.onmessage = (event) => onEvent(JSON.parse(event.data));
  ws.onerror = () => fail("WebSocket error");
  ws.onclose = () => {
    if (connected) hangup();
  };
}

function buildSessionConfig() {
  return {
    system_prompt: SESSION_CONFIG.systemPrompt,
    greeting: SESSION_CONFIG.greeting,
    input: SESSION_CONFIG.input,
    output: SESSION_CONFIG.output,
  };
}

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia(SESSION_CONFIG.microphone);
  micPipeline = AudioHelpers.createMicrophonePipeline(stream, {
    bufferSize: SESSION_CONFIG.processorBufferSize,
  });
  player = AudioHelpers.createPcmPlayer({ sampleRate: SESSION_CONFIG.outputSampleRate });
  await player.resume();
  await micPipeline.start((frame, sampleRate) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = AudioHelpers.downsampleToPCM(frame, sampleRate, SESSION_CONFIG.inputSampleRate);
    ws.send(JSON.stringify({ type: "input.audio", audio: AudioHelpers.bytesToB64(pcm) }));
  });

  connected = true;
  connBtn.textContent = "■ Hang up";
  connBtn.classList.add("live");
  setStatus("● Connected - just talk", "live");
}

function onEvent(event) {
  switch (event.type) {
    case "transcript.user":
      return addTurn("you", "You", event.text);
    case "transcript.agent":
      return addTurn("agent", "Agent", event.text);
    case "reply.audio":
      return player.playBase64Chunk(event.data);
    case "input.speech.started":
      return bargeIn();
    case "reply.done":
      if (event.status === "interrupted") bargeIn();
      return;
    case "session.error":
      return fail(event.message || "session error");
  }
}

function bargeIn() {
  if (player) player.stopQueuedAudio();
}

function addTurn(className, speaker, text) {
  if (!text) return;
  const turn = document.createElement("div");
  turn.className = "turn " + className;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = speaker + ": ";
  turn.append(who, document.createTextNode(text));
  logEl.appendChild(turn);
  turn.scrollIntoView({ block: "end" });
}

function hangup() {
  connected = false;
  connBtn.textContent = "● Connect";
  connBtn.classList.remove("live");
  setStatus("Disconnected", "idle");
  bargeIn();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  if (micPipeline) micPipeline.close();
  if (player) player.close();
  ws = null;
  micPipeline = null;
  player = null;
}

function fail(message) {
  setStatus("Error: " + message, "error");
  if (connected) hangup();
}
