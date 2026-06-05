import AVFoundation
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

let debugEnabled = ProcessInfo.processInfo.environment["AAI_MACOS_AUDIO_DEBUG"] == "1"

func debugLog(_ message: String) {
    if debugEnabled {
        fputs("[aai-macos-audio] \(message)\n", stderr)
    }
}

enum CaptureError: Error, CustomStringConvertible {
    case noDisplay
    case screenAudioPermissionDenied
    case conversionFailed(String)

    var description: String {
        switch self {
        case .noDisplay:
            return "No display is available for ScreenCaptureKit audio capture."
        case .screenAudioPermissionDenied:
            return "Screen & System Audio Recording permission is required for system audio capture."
        case .conversionFailed(let message):
            return message
        }
    }
}

struct Options {
    var sampleRate = 16_000
    var chunkFrames = 1_600

    static func parse() throws -> Options {
        var options = Options()
        var args = Array(CommandLine.arguments.dropFirst())
        while let arg = args.first {
            args.removeFirst()
            switch arg {
            case "--sample-rate":
                guard let value = args.first, let rate = Int(value), rate > 0 else {
                    throw CaptureError.conversionFailed("--sample-rate expects a positive integer.")
                }
                args.removeFirst()
                options.sampleRate = rate
            case "--chunk-frames":
                guard let value = args.first, let frames = Int(value), frames > 0 else {
                    throw CaptureError.conversionFailed("--chunk-frames expects a positive integer.")
                }
                args.removeFirst()
                options.chunkFrames = frames
            case "--system-only":
                continue
            default:
                throw CaptureError.conversionFailed("Unknown argument: \(arg)")
            }
        }
        return options
    }
}

final class SampleRing {
    private var samples: [Float] = []
    private let lock = NSLock()
    private let maxSamples: Int

    init(sampleRate: Int, maxSeconds: Double = 5.0) {
        self.maxSamples = max(1, Int(Double(sampleRate) * maxSeconds))
    }

    func append(_ newSamples: [Float]) {
        guard !newSamples.isEmpty else { return }
        lock.lock()
        samples.append(contentsOf: newSamples)
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
        lock.unlock()
    }

    func take(_ count: Int) -> [Float] {
        lock.lock()
        let available = min(count, samples.count)
        var output = [Float](repeating: 0, count: count)
        if available > 0 {
            output.replaceSubrange(0..<available, with: samples.prefix(available))
            samples.removeFirst(available)
        }
        lock.unlock()
        return output
    }
}

final class PCMConverter {
    private let outputFormat: AVAudioFormat

    init(sampleRate: Int) throws {
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw CaptureError.conversionFailed(
                "Could not create a \(sampleRate) Hz mono float audio format."
            )
        }
        self.outputFormat = outputFormat
    }

    func convert(_ input: AVAudioPCMBuffer) throws -> [Float] {
        if input.frameLength == 0 {
            return []
        }
        guard let converter = AVAudioConverter(from: input.format, to: outputFormat) else {
            throw CaptureError.conversionFailed("Could not create an audio converter.")
        }
        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up)) + 512
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            throw CaptureError.conversionFailed("Could not allocate a converted audio buffer.")
        }
        var suppliedInput = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            if suppliedInput {
                outStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            outStatus.pointee = .haveData
            return input
        }
        if status == .error {
            throw error ?? CaptureError.conversionFailed("Audio conversion failed.")
        }
        guard let channel = output.floatChannelData?[0] else {
            return []
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }
}

func audioFormat(for sampleBuffer: CMSampleBuffer) -> AVAudioFormat? {
    guard
        let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
        let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
    else {
        return nil
    }
    return AVAudioFormat(streamDescription: streamDescription)
}

func peak(_ samples: [Float]) -> Float {
    var result: Float = 0
    for sample in samples {
        result = max(result, abs(sample))
    }
    return result
}

final class Mixer {
    private let systemRing: SampleRing
    private let sampleRate: Int
    private let chunkFrames: Int
    private let queue = DispatchQueue(label: "aai.macos-audio.mixer", qos: .userInteractive)
    private var timer: DispatchSourceTimer?
    private var wroteFirstChunk = false

    init(sampleRate: Int, chunkFrames: Int) {
        self.systemRing = SampleRing(sampleRate: sampleRate)
        self.sampleRate = sampleRate
        self.chunkFrames = chunkFrames
    }

    func appendSystem(_ samples: [Float]) {
        systemRing.append(samples)
    }

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        let intervalNanoseconds = UInt64(Double(chunkFrames) / Double(sampleRate) * 1_000_000_000)
        timer.schedule(deadline: .now(), repeating: .nanoseconds(Int(intervalNanoseconds)))
        timer.setEventHandler { [weak self] in
            self?.writeChunk()
        }
        self.timer = timer
        timer.resume()
    }

    private func writeChunk() {
        let system = systemRing.take(chunkFrames)
        var data = Data(capacity: chunkFrames * 2)
        for index in 0..<chunkFrames {
            let clamped = max(-1.0, min(1.0, system[index]))
            let sample = Int16(max(Double(Int16.min), min(Double(Int16.max), Double(clamped) * 32767.0)))
            var littleEndian = sample.littleEndian
            withUnsafeBytes(of: &littleEndian) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        if !wroteFirstChunk {
            debugLog("writing first PCM chunk")
            wroteFirstChunk = true
        }
        writeStdout(data)
    }

    private func writeStdout(_ data: Data) {
        data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < rawBuffer.count {
                let written = Darwin.write(
                    STDOUT_FILENO,
                    baseAddress.advanced(by: offset),
                    rawBuffer.count - offset
                )
                if written <= 0 {
                    exit(0)
                }
                offset += written
            }
        }
    }
}

final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let mixer: Mixer
    private let converter: PCMConverter
    private let queue = DispatchQueue(label: "aai.macos-audio.system", qos: .userInteractive)
    private var stream: SCStream?
    private var sawSystemAudio = false
    private var loggedSystemLevel = false

    init(mixer: Mixer, sampleRate: Int) throws {
        self.mixer = mixer
        self.converter = try PCMConverter(sampleRate: sampleRate)
    }

    func start(sampleRate: Int) async throws {
        debugLog("loading shareable content")
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )
        debugLog("shareable content loaded: displays=\(content.displays.count)")
        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let configuration = SCStreamConfiguration()
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        configuration.queueDepth = 3
        configuration.showsCursor = false
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = sampleRate
        configuration.channelCount = 1

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        debugLog("adding audio output")
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        debugLog("starting ScreenCaptureKit capture")
        try await stream.startCapture()
        debugLog("ScreenCaptureKit capture started")
        self.stream = stream
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio, sampleBuffer.isValid else {
            return
        }
        handleSystem(sampleBuffer)
    }

    private func handleSystem(_ sampleBuffer: CMSampleBuffer) {
        if !sawSystemAudio {
            debugLog("received first system audio sample")
            sawSystemAudio = true
        }
        do {
            try sampleBuffer.withAudioBufferList { audioBufferList, _ in
                guard
                    let format = audioFormat(for: sampleBuffer),
                    let pcmBuffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        bufferListNoCopy: audioBufferList.unsafePointer
                    )
                else {
                    return
                }
                let samples = try converter.convert(pcmBuffer)
                if !loggedSystemLevel {
                    debugLog("system peak=\(peak(samples)) samples=\(samples.count)")
                    loggedSystemLevel = true
                }
                mixer.appendSystem(samples)
            }
        } catch {
            fputs("System audio conversion failed: \(error)\n", stderr)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("ScreenCaptureKit stopped: \(error)\n", stderr)
        exit(1)
    }
}

@main
struct Main {
    static func main() async {
        do {
            debugLog("helper starting")
            let options = try Options.parse()
            debugLog(
                "parsed options: sampleRate=\(options.sampleRate) "
                    + "chunkFrames=\(options.chunkFrames)"
            )
            let mixer = Mixer(
                sampleRate: options.sampleRate,
                chunkFrames: options.chunkFrames
            )
            let system = try SystemAudioCapture(mixer: mixer, sampleRate: options.sampleRate)

            try await system.start(sampleRate: options.sampleRate)
            debugLog("starting mixer")
            mixer.start()
            while true {
                _ = system
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
    }
}
