function createMicrophonePipeline(stream, options = {}) {
  const bufferSize = options.bufferSize || 4096;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContextClass();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

  return {
    audioCtx,
    async start(onFrame) {
      await audioCtx.resume();
      source.connect(processor);
      processor.connect(audioCtx.destination);
      processor.onaudioprocess = (event) => {
        onFrame(event.inputBuffer.getChannelData(0), audioCtx.sampleRate);
      };
    },
    close() {
      processor.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      audioCtx.close();
    },
  };
}

function createPcmPlayer(options = {}) {
  const sampleRate = options.sampleRate || 24000;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const playCtx = new AudioContextClass({ sampleRate });
  let playHead = 0;
  let sources = [];

  return {
    async resume() {
      await playCtx.resume();
    },
    playBase64Chunk(base64Audio) {
      const int16 = b64ToInt16(base64Audio);
      const buffer = playCtx.createBuffer(1, int16.length, sampleRate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < int16.length; i++) channel[i] = int16[i] / 0x8000;

      const source = playCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(playCtx.destination);
      const startAt = Math.max(playCtx.currentTime, playHead);
      source.start(startAt);
      playHead = startAt + buffer.duration;
      sources.push(source);
      source.onended = () => {
        sources = sources.filter((item) => item !== source);
      };
    },
    stopQueuedAudio() {
      sources.forEach((source) => {
        try {
          source.stop();
        } catch (_) {}
      });
      sources = [];
      playHead = 0;
    },
    close() {
      this.stopQueuedAudio();
      playCtx.close();
    },
  };
}

function downsampleToPCM(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const sample = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output.buffer;
}

function bytesToB64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i++)
    binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function b64ToInt16(base64Audio) {
  const binary = atob(base64Audio);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

window.AudioHelpers = {
  createMicrophonePipeline,
  createPcmPlayer,
  downsampleToPCM,
  bytesToB64,
};
