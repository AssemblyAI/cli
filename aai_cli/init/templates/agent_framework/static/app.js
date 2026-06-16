const SESSION_CONFIG = {
  inputSampleRate: 16000,
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

connBtn.addEventListener("click", () =>
  connected ? hangup() : connect().catch(fail),
);

function setStatus(message, state) {
  statusEl.textContent = message;
  statusEl.dataset.state = state;
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

async function connect() {
  setStatus("Connecting...", "idle");
  ws = new WebSocket(wsUrl());
  ws.onopen = () => startMic().catch(fail);
  ws.onmessage = (event) => onEvent(JSON.parse(event.data));
  ws.onerror = () => fail("WebSocket error");
  ws.onclose = () => {
    if (connected) hangup();
  };
}

async function startMic() {
  // Create the player first: the server speaks the greeting the instant the
  // socket opens, so `reply.audio` can arrive before getUserMedia's permission
  // prompt resolves. Setting `player` synchronously here (before the first
  // await) guarantees it exists when onEvent handles that first audio frame.
  player = AudioHelpers.createPcmPlayer({
    sampleRate: SESSION_CONFIG.outputSampleRate,
  });
  await player.resume();
  const stream = await navigator.mediaDevices.getUserMedia(
    SESSION_CONFIG.microphone,
  );
  micPipeline = AudioHelpers.createMicrophonePipeline(stream, {
    bufferSize: SESSION_CONFIG.processorBufferSize,
  });
  await micPipeline.start((frame, sampleRate) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = AudioHelpers.downsampleToPCM(
      frame,
      sampleRate,
      SESSION_CONFIG.inputSampleRate,
    );
    ws.send(
      JSON.stringify({
        type: "input.audio",
        audio: AudioHelpers.bytesToB64(pcm),
      }),
    );
  });

  connected = true;
  connBtn.textContent = "■ Hang up";
  connBtn.dataset.state = "connected";
  setStatus("● Connected - just talk", "live");
}

function onEvent(event) {
  switch (event.type) {
    case "transcript.user":
      return addTurn("you", "You", event.text);
    case "transcript.agent":
      return addTurn("agent", "Agent", event.text);
    case "reply.audio":
      if (player) player.playBase64Chunk(event.data);
      return;
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

function addTurn(speakerKind, speaker, text) {
  if (!text) return;
  const turn = document.createElement("div");
  turn.className = "conversation-turn";
  turn.dataset.speaker = speakerKind;
  const who = document.createElement("span");
  who.className = "turn-speaker";
  who.textContent = speaker + ": ";
  turn.append(who, document.createTextNode(text));
  logEl.appendChild(turn);
  turn.scrollIntoView({ block: "end" });
}

function hangup() {
  connected = false;
  connBtn.textContent = "● Connect";
  connBtn.dataset.state = "idle";
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
