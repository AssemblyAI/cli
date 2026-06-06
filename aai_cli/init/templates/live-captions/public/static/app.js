const STREAMING_CONFIG = {
  sampleRate: 16000,
  encoding: "pcm_s16le",
  speechModel: "u3-rt-pro",
  formatTurns: true,
  processorBufferSize: 4096,
};

const recBtn = document.getElementById("rec");
const statusEl = document.getElementById("status");
const finalEl = document.getElementById("final");
const partialEl = document.getElementById("partial");

let ws = null;
let audioPipeline = null;
let recording = false;

recBtn.addEventListener("click", () =>
  recording ? stop() : start().catch(fail),
);

function setStatus(message, state) {
  statusEl.textContent = message;
  statusEl.dataset.state = state;
}

async function start() {
  setStatus("Connecting...", "idle");
  const res = await fetch("/api/token", { method: "POST" });
  if (!res.ok) return fail(await res.text());
  const { token, ws_url } = await res.json();

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioPipeline = AudioHelpers.createMicrophonePipeline(stream, {
    bufferSize: STREAMING_CONFIG.processorBufferSize,
  });

  const params = new URLSearchParams({
    token,
    sample_rate: String(STREAMING_CONFIG.sampleRate),
    encoding: STREAMING_CONFIG.encoding,
    speech_model: STREAMING_CONFIG.speechModel,
    format_turns: String(STREAMING_CONFIG.formatTurns),
  });
  ws = new WebSocket(`${ws_url}?${params}`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    recording = true;
    recBtn.textContent = "■ Stop";
    recBtn.dataset.state = "recording";
    setStatus("● Live", "live");
    audioPipeline.connect((frame, sampleRate) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          AudioHelpers.downsampleToPCM(
            frame,
            sampleRate,
            STREAMING_CONFIG.sampleRate,
          ),
        );
      }
    });
  };
  ws.onmessage = (event) => onMessage(JSON.parse(event.data));
  ws.onerror = () => fail("WebSocket error");
  ws.onclose = () => {
    if (recording) stop();
  };
}

function onMessage(message) {
  if (message.type !== "Turn") return;
  if (message.end_of_turn) {
    finalEl.textContent += (message.transcript || "") + " ";
    partialEl.textContent = "";
  } else {
    partialEl.textContent = message.transcript || "";
  }
}

function stop() {
  recording = false;
  recBtn.textContent = "● Record";
  recBtn.dataset.state = "idle";
  setStatus("Stopped", "idle");
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "Terminate" }));
  if (audioPipeline) audioPipeline.close();
  ws = null;
  audioPipeline = null;
}

function fail(message) {
  setStatus("Error: " + message, "error");
  if (recording) stop();
}
