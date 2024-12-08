import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Vocos
import AVFoundation

// MARK: - F5TTS

public class F5TTS: Module {
    enum F5TTSError: Error {
        case unableToLoadModel
        case unableToLoadReferenceAudio
        case unableToDetermineDuration
    }

    static let targetRMS: Float = 0.1

    public let melSpec: MelSpec
    public let transformer: DiT

    let dim: Int
    let numChannels: Int
    let vocabCharMap: [String: Int]
    let _durationPredictor: DurationPredictor?

    init(
        transformer: DiT,
        melSpec: MelSpec,
        vocabCharMap: [String: Int],
        durationPredictor: DurationPredictor? = nil
    ) {
        self.melSpec = melSpec
        self.numChannels = self.melSpec.nMels
        self.transformer = transformer
        self.dim = transformer.dim
        self.vocabCharMap = vocabCharMap
        self._durationPredictor = durationPredictor

        super.init()
    }

    private func odeint(fun: (Float, MLXArray) -> MLXArray, y0: MLXArray, t: MLXArray) -> MLXArray {
        var ys = [y0]
        var yCurrent = y0

        for i in 0..<(t.shape[0] - 1) {
            let tCurrent = t[i].item(Float.self)
            let dt = t[i + 1].item(Float.self) - tCurrent

            let k1 = fun(tCurrent, yCurrent)
            let mid = yCurrent + 0.5 * dt * k1

            let k2 = fun(tCurrent + 0.5 * dt, mid)
            let yNext = yCurrent + dt * k2

            ys.append(yNext)
            yCurrent = yNext
        }

        return MLX.stacked(ys, axis: 0)
    }

    private func sample(
        cond: MLXArray,
        text: [String],
        duration: Int? = nil,
        lens: MLXArray? = nil,
        steps: Int = 32,
        cfgStrength: Double = 2.0,
        swayCoef: Double? = -1.0,
        seed: Int? = nil,
        maxDuration: Int = 4096,
        vocoder: ((MLXArray) -> MLXArray)? = nil,
        noRefAudio: Bool = false,
        editMask: MLXArray? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) throws -> (MLXArray, MLXArray) {
        MLX.eval(self.parameters())

        var cond = cond

        // raw wave

        if cond.ndim == 2 {
            cond = cond.reshaped([cond.shape[1]])
            cond = self.melSpec(x: cond)
        }

        let batch = cond.shape[0]
        let condSeqLen = cond.shape[1]
        var lens = lens ?? MLX.full([batch], values: condSeqLen, type: Int.self)

        // text

        let inputText = listStrToIdx(text, vocabCharMap: vocabCharMap)
        let textLens = (inputText .!= -1).sum(axis: -1)
        lens = MLX.maximum(textLens, lens)

        var condMask = lensToMask(t: lens)
        if let editMask = editMask {
            condMask = condMask & editMask
        }

        // duration
        var resolvedDuration: MLXArray? = (duration != nil) ? MLXArray(duration!) : nil

        if resolvedDuration == nil, let durationPredictor = self._durationPredictor {
            let estimatedDurationInSeconds = durationPredictor(cond, text: text).item(Float32.self)
            resolvedDuration = MLXArray(Int(Double(estimatedDurationInSeconds) * F5TTS.framesPerSecond))
        }

        guard let resolvedDuration else {
            throw F5TTSError.unableToDetermineDuration
        }

        print("Generating \(Double(resolvedDuration.item(Float32.self)) / F5TTS.framesPerSecond) seconds of audio...")

        var duration = resolvedDuration
        duration = MLX.clip(MLX.maximum(lens + 1, duration), min: 0, max: maxDuration)
        let maxDuration = duration.max().item(Int.self)

        cond = MLX.padded(cond, widths: [.init((0, 0)), .init((0, maxDuration - condSeqLen)), .init((0, 0))])
        condMask = MLX.padded(condMask, widths: [.init((0, 0)), .init((0, maxDuration - condMask.shape[1]))], value: MLXArray(false))
        condMask = condMask.expandedDimensions(axis: -1)
        let stepCond = MLX.where(condMask, cond, MLX.zeros(like: cond))

        let mask: MLXArray? = (batch > 1) ? lensToMask(t: duration) : nil

        if noRefAudio {
            cond = MLX.zeros(like: cond)
        }

        // neural ode

        let fn: (Float, MLXArray) -> MLXArray = { t, x in
            let pred = self.transformer(
                x: x,
                cond: stepCond,
                text: inputText,
                time: MLXArray(t),
                dropAudioCond: false,
                dropText: false,
                mask: mask
            )

            guard cfgStrength > 1e-5 else {
                pred.eval()
                return pred
            }

            let nullPred = self.transformer(
                x: x,
                cond: stepCond,
                text: inputText,
                time: MLXArray(t),
                dropAudioCond: true,
                dropText: true,
                mask: mask
            )

            progressHandler?(Double(t))

            let output = pred + (pred - nullPred) * cfgStrength
            output.eval()

            return output
        }

        // noise input

        var y0: [MLXArray] = []
        for dur in duration {
            if let seed = seed {
                MLXRandom.seed(UInt64(seed))
            }
            let noise = MLXRandom.normal([dur.item(Int.self), self.numChannels])
            y0.append(noise)
        }
        let y0Padded = padSequence(y0, paddingValue: 0.0)

        var t = MLXArray.linspace(Float32(0.0), Float32(1.0), count: steps)

        if let coef = swayCoef {
            t = t + coef * (MLX.cos(MLXArray(.pi) / 2 * t) - 1 + t)
        }

        let trajectory = self.odeint(fun: fn, y0: y0Padded, t: t)
        let sampled = trajectory[-1]
        var out = MLX.where(condMask, cond, sampled)

        if let vocoder = vocoder {
            out = vocoder(out)
        }
        out.eval()

        return (out, trajectory)
    }

    public func generate(
        text: String,
        referenceAudioURL: URL? = nil,
        referenceAudioText: String? = nil,
        duration: TimeInterval? = nil,
        cfg: Double = 2.0,
        sway: Double = -1.0,
        speed: Double = 1.0,
        seed: Int? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MLXArray {
        print("Loading Vocos model...")
        let vocos = try await Vocos.fromPretrained(repoId: "lucasnewman/vocos-mel-24khz-mlx")

        // load the reference audio + text

        var audio: MLXArray
        let referenceText: String

        if let referenceAudioURL {
            audio = try F5TTS.loadAudioArray(url: referenceAudioURL)
            referenceText = referenceAudioText ?? ""
        } else {
            let refAudioAndCaption = try F5TTS.referenceAudio()
            (audio, referenceText) = refAudioAndCaption
        }

        let refAudioDuration = Double(audio.shape[0]) / Double(F5TTS.sampleRate)
        print("Using reference audio with duration: \(refAudioDuration)")

        // generate the audio

        let normalizedAudio = F5TTS.normalizeAudio(audio: audio)
        let processedText = referenceText + " " + text

        let (outputAudio, _) = try self.sample(
            cond: normalizedAudio.expandedDimensions(axis: 0),
            text: [processedText],
            duration: nil,
            steps: 32,
            cfgStrength: cfg,
            swayCoef: sway,
            seed: seed,
            vocoder: vocos.decode
        ) { progress in
            print("Generation progress: \(progress)")
            progressHandler?(progress)
        }

        return outputAudio[audio.shape[0]...]
    }
}

// MARK: - Pretrained Models

public extension F5TTS {
    static func fromPretrained(repoId: String, bit: Int? = nil, downloadProgress: ((Progress) -> Void)? = nil) async throws -> F5TTS {
        // Determine bit width from model name if not explicitly provided
        var resolvedBit = bit
        if resolvedBit == nil {
            if repoId.contains("8bit") {
                print("Loading model with 8bit quantization")
                resolvedBit = 8
            } else if repoId.contains("4bit") {
                print("Loading model with 4bit quantization")
                resolvedBit = 4
            }
        }
        
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, matching: ["*.safetensors", "*.txt"]) { progress in
            downloadProgress?(progress)
        }
        
        let modelURL = modelDirectoryURL.appendingPathComponent("model.safetensors")
        let modelWeights = try loadArrays(url: modelURL)

        // mel spec
        guard let filterbankURL = Bundle.module.url(forResource: "mel_filters", withExtension: "npy") else {
            throw F5TTSError.unableToLoadModel
        }
        let filterbank = try MLX.loadArray(url: filterbankURL)

        // vocab
        let vocabURL = modelDirectoryURL.appendingPathComponent("vocab.txt")
        guard let vocabString = try String(data: Data(contentsOf: vocabURL), encoding: .utf8) else {
            throw F5TTSError.unableToLoadModel
        }

        let vocabEntries = vocabString.split(separator: "\n").map { String($0) }
        let vocab = Dictionary(uniqueKeysWithValues: zip(vocabEntries, vocabEntries.indices))

        // duration model
        var durationPredictor: DurationPredictor?
        let durationModelURL = modelDirectoryURL.appendingPathComponent("duration_v2.safetensors")
        do {
            let durationModelWeights = try loadArrays(url: durationModelURL)

            let durationTransformer = DurationTransformer(
                dim: 512,
                depth: 8,
                heads: 8,
                dimHead: 64,
                ffMult: 2,
                textNumEmbeds: vocab.count,
                textDim: 512,
                convLayers: 2
            )
            let predictor = DurationPredictor(
                transformer: durationTransformer,
                melSpec: MelSpec(filterbank: filterbank),
                vocabCharMap: vocab
            )
            try predictor.update(parameters: ModuleParameters.unflattened(durationModelWeights), verify: [.all])
            durationPredictor = predictor
          
        } catch {
            print("Warning: no duration predictor model found: \(error)")
        }

        // model
        let dit = DiT(
            dim: 1024,
            depth: 22,
            heads: 16,
            ffMult: 2,
            textNumEmbeds: vocab.count,
            textDim: 512,
            convLayers: 4
        )
        let f5tts = F5TTS(
            transformer: dit,
            melSpec: MelSpec(filterbank: filterbank),
            vocabCharMap: vocab,
            durationPredictor: durationPredictor
        )
        
        // Apply quantization if requested
        if let bit = resolvedBit {
            if bit == 4 || bit == 8 {
                print("Loading model with \(bit)bit quantization")
                // Quantize all Linear layers with input dimension divisible by 64
                var quantizedWeights: [String: MLXArray] = [:]
                
                for (path, param) in f5tts.parameters() {
                    if path.hasSuffix("weight"),
                       let array = param as? MLXArray,
                       array.shape[1] % 64 == 0 {
                        let (wq, _, _) = MLX.quantized(array, bits: bit)
                        quantizedWeights[path] = wq
                    }
                }
                
                // Update the model with quantized weights
                if !quantizedWeights.isEmpty {
                    do {
                        try f5tts.update(parameters: ModuleParameters.unflattened(quantizedWeights))
                    } catch {
                        print("Warning: Failed to update quantized weights: \(error)")
                    }
                }
            } else {
                print("Warning: Unsupported bit width \(bit). Skipping quantization.")
            }
        }

        try f5tts.update(parameters: ModuleParameters.unflattened(modelWeights), verify: [.all])
        return f5tts
    }
}

// MARK: - Utilities

public extension F5TTS {
    static var sampleRate: Int = 24000
    static var hopLength: Int = 256
    static var framesPerSecond: Double = .init(sampleRate) / Double(hopLength)

    static func loadAudioArray(url: URL) throws -> MLXArray {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = UInt32(audioFile.length)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        try audioFile.read(into: buffer)
        
        if let floatData = buffer.floatChannelData?[0] {
            let data = Array(UnsafeBufferPointer(start: floatData, count: Int(frameCount)))
            return MLXArray(data)
        }
        throw F5TTSError.unableToLoadReferenceAudio
    }

    static func referenceAudio() throws -> (MLXArray, String) {
        guard let url = Bundle.module.url(forResource: "test_en_1_ref_short", withExtension: "wav") else {
            throw F5TTSError.unableToLoadReferenceAudio
        }
        let audio = try loadAudioArray(url: url)
        return (audio, "Some call me nature, others call me mother nature.")
    }

    static func normalizeAudio(audio: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(MLX.square(audio)))
        if rms.item(Float.self) < targetRMS {
            return audio * (targetRMS / rms)
        }
        return audio
    }

    static func estimateDuration(
        text: String,
        referenceAudioDuration: TimeInterval,
        referenceText: String,
        speed: Double = 1.0
    ) -> TimeInterval {
        let refDurationInFrames = Int(referenceAudioDuration * framesPerSecond)
        let refTextLength = Double(referenceText.count)
        let genTextLength = Double(text.count)
        
        let refAudioToTextRatio = Double(refDurationInFrames) / refTextLength
        let textLength = genTextLength / speed
        let estimatedDurationInFrames = Int(refAudioToTextRatio * textLength)
        
        let estimatedDuration = TimeInterval(estimatedDurationInFrames) / Self.framesPerSecond
        print("Using duration of \(estimatedDuration) seconds (\(estimatedDurationInFrames) frames) for generated speech.")
        
        return estimatedDuration
    }
}

// MLX utilities

func lensToMask(t: MLXArray, length: Int? = nil) -> MLXArray {
    let maxLength = length ?? t.max(keepDims: false).item(Int.self)
    let seq = MLXArray(0..<maxLength)
    let expandedSeq = seq.expandedDimensions(axis: 0)
    let expandedT = t.expandedDimensions(axis: 1)
    return MLX.less(expandedSeq, expandedT)
}

func padToLength(_ t: MLXArray, length: Int, value: Float? = nil) -> MLXArray {
    let ndim = t.ndim
    
    guard let seqLen = t.shape.last, length > seqLen else {
        return t[0..., .ellipsis]
    }
    
    let paddingValue = MLXArray(value ?? 0.0)
    
    if ndim == 1 {
        return MLX.padded(t, widths: [.init((0, length - seqLen))], value: paddingValue)
    } else if ndim == 2 {
        return MLX.padded(t, widths: [.init((0, 0)), .init((0, length - seqLen))], value: paddingValue)
    } else {
        fatalError("Unsupported padding dims: \(ndim)")
    }
}

func padSequence(_ t: [MLXArray], paddingValue: Float = 0.0) -> MLXArray {
    let maxLen = t.map { $0.shape[0] }.max() ?? 0
    let padded = t.map { padToLength($0, length: maxLen, value: paddingValue) }
    return MLX.stacked(padded, axis: 0)
}

func listStrToIdx(_ text: [String], vocabCharMap: [String: Int]) -> MLXArray {
    let listIdxTensors = text.map { str -> [Int] in
        str.map { char in
            vocabCharMap[String(char)] ?? 0
        }
    }
    
    let maxLen = listIdxTensors.map { $0.count }.max() ?? 0
    let padded = listIdxTensors.map { idxs -> [Int] in
        idxs + Array(repeating: -1, count: maxLen - idxs.count)
    }
    
    return MLXArray(padded.flatMap { $0 }).reshaped([text.count, maxLen])
}

extension F5TTS {
    public func quantize(bits: Int = 4) {
        // Quantize transformer components
        
        // Input embedding linear layer
        quantizeLinear(transformer.input_embed.proj, bits: bits)
        
        // Transformer blocks
        for block in transformer.transformer_blocks {
            // Attention components
            quantizeLinear(block.attn.to_q, bits: bits)
            quantizeLinear(block.attn.to_k, bits: bits)
            quantizeLinear(block.attn.to_v, bits: bits)
            if let toOutLinear = block.attn.to_out.layers.first as? Linear {
                quantizeLinear(toOutLinear, bits: bits)
            }
            
            // Feed forward components
            if let ff = block.ff.ff.layers.first as? Sequential,
               let ffLinear = ff.layers.first as? Linear {
                quantizeLinear(ffLinear, bits: bits)
            }
            if let ffOutLinear = block.ff.ff.layers.last as? Linear {
                quantizeLinear(ffOutLinear, bits: bits)
            }
        }
        
        // Final projection
        quantizeLinear(transformer.proj_out, bits: bits)
        
        // Duration predictor if present
        if let durationPredictor = _durationPredictor {
            // Quantize duration predictor components
            quantizeLinear(durationPredictor.transformer.input_embed.proj, bits: bits)
            
            for block in durationPredictor.transformer.transformer_blocks {
                quantizeLinear(block.attn.to_q, bits: bits)
                quantizeLinear(block.attn.to_k, bits: bits)
                quantizeLinear(block.attn.to_v, bits: bits)
                if let toOutLinear = block.attn.to_out.layers.first as? Linear {
                    quantizeLinear(toOutLinear, bits: bits)
                }
                
                if let ff = block.ff.ff.layers.first as? Sequential,
                   let ffLinear = ff.layers.first as? Linear {
                    quantizeLinear(ffLinear, bits: bits)
                }
                if let ffOutLinear = block.ff.ff.layers.last as? Linear {
                    quantizeLinear(ffOutLinear, bits: bits)
                }
            }
            
            if let toPredLinear = durationPredictor.to_pred.layers.first as? Linear {
                quantizeLinear(toPredLinear, bits: bits)
            }
        }
    }
    
    private func quantizeLinear(_ linear: Linear, bits: Int) {
        let weight = linear.weight
        let quantization = MLX.quantized(weight, bits: bits)
        linear.weight = quantization.wq
        linear.scales = quantization.scales
        linear.biases = quantization.biases
        
        if let bias = linear.bias {
            let biasQuantization = MLX.quantized(bias, bits: bits)
            linear.bias = biasQuantization.wq
        }
    }
}
