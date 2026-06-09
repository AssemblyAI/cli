function createMicrophonePipeline(stream, options = {}) {
  const bufferSize = options.bufferSize || 4096;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContextClass();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

  return {
    audioCtx,
    connect(onFrame) {
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

window.AudioHelpers = {
  createMicrophonePipeline,
  downsampleToPCM,
};
