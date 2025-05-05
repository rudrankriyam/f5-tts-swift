import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Vocos

// MARK: - F5TTS

func odeint_euler(fun: (Float, MLXArray) -> MLXArray, y0: MLXArray, t: MLXArray) -> MLXArray {
  var ys = [y0]
  var yCurrent = y0

  for i in 0..<(t.shape[0] - 1) {
    let tCurrent = t[i].item(Float.self)
    let dt = t[i + 1].item(Float.self) - tCurrent

    let k = fun(tCurrent, yCurrent)
    let yNext = yCurrent + dt * k

    ys.append(yNext)
    yCurrent = yNext
  }

  return MLX.stacked(ys, axis: 0)
}

func odeint_midpoint(fun: (Float, MLXArray) -> MLXArray, y0: MLXArray, t: MLXArray) -> MLXArray {
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

func odeint_rk4(fun: (Float, MLXArray) -> MLXArray, y0: MLXArray, t: MLXArray) -> MLXArray {
  var ys = [y0]
  var yCurrent = y0

  for i in 0..<(t.shape[0] - 1) {
    let tCurrent = t[i].item(Float.self)
    let dt = t[i + 1].item(Float.self) - tCurrent

    let k1 = fun(tCurrent, yCurrent)
    let k2 = fun(tCurrent + 0.5 * dt, yCurrent + 0.5 * dt * k1)
    let k3 = fun(tCurrent + 0.5 * dt, yCurrent + 0.5 * dt * k2)
    let k4 = fun(tCurrent + dt, yCurrent + dt * k3)

    let yNext = yCurrent + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    ys.append(yNext)
    yCurrent = yNext
  }

  return MLX.stacked(ys)
}

public class F5TTS: Module {
  public enum ODEMethod: String {
    case euler
    case midpoint
    case rk4
  }

  enum F5TTSError: Error {
    case unableToLoadModel
    case unableToLoadReferenceAudio
    case unableToDetermineDuration
  }

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

  private func sample(
    cond: MLXArray,
    text: [String],
    duration: Int? = nil,
    lens: MLXArray? = nil,
    steps: Int = 8,
    method: ODEMethod = .rk4,
    cfgStrength: Double = 2.0,
    swayCoef: Double? = -1.0,
    seed: Int? = nil,
    maxDuration: Int = 4096,
    vocoder: ((MLXArray) -> MLXArray)? = nil,
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

    // duration
    var resolvedDuration: MLXArray? = (duration != nil) ? MLXArray(duration!) : nil

    if resolvedDuration == nil, let durationPredictor = self._durationPredictor {
      let estimatedDurationInSeconds = durationPredictor(cond, text: text).item(Float32.self)
      resolvedDuration = MLXArray(Int(Double(estimatedDurationInSeconds) * F5TTS.framesPerSecond))
    }

    guard let resolvedDuration else {
      throw F5TTSError.unableToDetermineDuration
    }

    print(
      "Generating \(Double(resolvedDuration.item(Float32.self)) / F5TTS.framesPerSecond) seconds of audio..."
    )

    var duration = resolvedDuration
    duration = MLX.clip(MLX.maximum(lens + 1, duration), min: 0, max: maxDuration)
    let maxDuration = duration.max().item(Int.self)

    cond = MLX.padded(
      cond, widths: [.init((0, 0)), .init((0, maxDuration - condSeqLen)), .init((0, 0))])
    condMask = MLX.padded(
      condMask, widths: [.init((0, 0)), .init((0, maxDuration - condMask.shape[1]))],
      value: MLXArray(false))
    condMask = condMask.expandedDimensions(axis: -1)
    let stepCond = MLX.where(condMask, cond, MLX.zeros(like: cond))

    let mask: MLXArray? = (batch > 1) ? lensToMask(t: duration) : nil

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
      if let seed {
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

    let odeintFn =
      switch method {
      case .euler: odeint_euler
      case .midpoint: odeint_midpoint
      case .rk4: odeint_rk4
      }

    let trajectory = odeintFn(fn, y0Padded, t)
    let sampled = trajectory[-1]
    var out = MLX.where(condMask, cond, sampled)

    if let vocoder {
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
    steps: Int = 8,
    method: ODEMethod = .rk4,
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
      steps: steps,
      method: method,
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

extension F5TTS {
  public static func fromPretrained(
    repoId: String, quantizationBits: Int? = nil, downloadProgress: ((Progress) -> Void)? = nil
  )
    async throws -> F5TTS
  {
    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId, matching: ["*.safetensors", "*.txt"]
    ) { progress in
      downloadProgress?(progress)
    }
    return try self.fromPretrained(
      modelDirectoryURL: modelDirectoryURL, quantizationBits: quantizationBits)
  }

  public static func fromPretrained(modelDirectoryURL: URL, quantizationBits: Int? = nil) throws
    -> F5TTS
  {
    // Determine model filename based on quantization
    var modelFilename = "model_v1.safetensors"
    if let bits = quantizationBits {
      modelFilename = "model_v1_\(bits)b.safetensors"
    }
    let modelURL = modelDirectoryURL.appendingPathComponent(modelFilename)

    // Always load these base components
    guard let filterbankURL = Bundle.module.url(forResource: "mel_filters", withExtension: "npy")
    else { throw F5TTSError.unableToLoadModel }
    let filterbank = try MLX.loadArray(url: filterbankURL)
    let vocabURL = modelDirectoryURL.appendingPathComponent("vocab.txt")
    guard let vocabString = try String(data: Data(contentsOf: vocabURL), encoding: .utf8) else {
      throw F5TTSError.unableToLoadModel
    }
    let vocabEntries = vocabString.split(separator: "\n").map { String($0) }
    let vocab = Dictionary(uniqueKeysWithValues: zip(vocabEntries, vocabEntries.indices))

    // --- Helper function for weight conversion ---
    func convertWeights(_ originalWeights: [String: MLXArray]) -> [String: MLXArray] {
      print("Converting weights...")
      var finalWeights: [String: MLXArray] = [:]
      for (k, v) in originalWeights {
        let originalKey = k  // Store original key for debugging
        var key = k.replacingOccurrences(of: "ema_model.", with: "")  // Remove potential prefix
        var value = v

        // Skip unnecessary keys
        if key.isEmpty || key.contains("mel_spec.") || ["initted", "step"].contains(key) {
          continue
        }

        // --- Key Renaming for Refactored Sequential Modules ---
        let keyBeforeRenaming = key  // Debug: Store key before specific renames

        // FeedForward (ff -> linear1, activation, dropout, linear2)
        key = key.replacingOccurrences(of: ".ff.ff.layers.0.layers.0.", with: ".ff.linear1.")  // Maps projectIn linear
        key = key.replacingOccurrences(of: ".ff.ff.layers.2.", with: ".ff.linear2.")  // Maps final linear

        // TimestepEmbedding (time_mlp -> time_mlp_linear1, time_mlp_activation, time_mlp_linear2)
        key = key.replacingOccurrences(of: ".time_mlp.layers.0.", with: ".time_mlp_linear1.")
        key = key.replacingOccurrences(of: ".time_mlp.layers.2.", with: ".time_mlp_linear2.")

        // Attention (to_out -> to_out_linear, to_out_dropout)
        key = key.replacingOccurrences(of: ".to_out.layers.0.", with: ".to_out_linear.")

        // ConvPositionEmbedding (conv1d -> conv1, act1, conv2, act2)
        key = key.replacingOccurrences(of: ".conv1d.layers.0.", with: ".conv1.")
        key = key.replacingOccurrences(of: ".conv1d.layers.2.", with: ".conv2.")

        // TextEmbedding (text_blocks: Sequential -> [ConvNeXtV2Block])
        // This replaces .text_blocks.layers.N. with .text_blocks.N. PRESERVING the prefix
        if let range = key.range(of: ".text_blocks.layers.") {
          let suffix = key[range.upperBound...]
          if let numberEndIndex = suffix.firstIndex(of: ".") {
            let numberString = String(suffix[..<numberEndIndex])
            let restOfKey = suffix[numberEndIndex...]
            let prefix = key[..<range.lowerBound]  // Get the part before ".text_blocks..."
            key = "\(prefix).text_blocks.\(numberString)\(restOfKey)"
          }  // else: format unexpected
        }

        // DurationPredictor (to_pred: Sequential -> PredictorHead)
        // Rename old sequential layer key to the new nested linear layer key
        if key.hasPrefix("to_pred.layers.0.") {
          key = key.replacingOccurrences(
            of: "to_pred.layers.0.", with: "to_pred.linear.", options: .anchored, range: nil)
        } else {
          // Check for keys within the duration predictor path if needed (less likely now)
          key = key.replacingOccurrences(of: ".to_pred.layers.0.", with: ".to_pred.linear.")
        }

        // --- Original Transpositions (Keep these) ---
        var didTranspose = false  // Debug flag
        let originalShape = value.shape  // Debug: Store original shape

        if key.hasSuffix(".dwconv.weight") {
          // Debug: Print before transposition for dwconv
          print("DEBUG (dwconv): Transposing key: \(key), Original Shape: \(originalShape)")
          value = value.transposed(0, 2, 1)
          didTranspose = true
          // Debug: Print after transposition
          print("DEBUG (dwconv): Transposed key: \(key), New Shape: \(value.shape)")
        }
        // Remove transposition for .conv1.weight and .conv2.weight as they are already correct
        /* else if key.hasSuffix(".conv1.weight") ... */

        // Note: Removed transposition check for .dwconv.bias as it wasn't doing anything.

        // Debug: Print original and final keys
        if originalKey != key {  // Only print if a change occurred
          print("DEBUG: Key Renamed: \'\(originalKey)\' -> \'\(key)\'")
        } else if keyBeforeRenaming != key {  // Print if specific renames happened
          print("DEBUG: Key Renamed (Specific): \'\(keyBeforeRenaming)\' -> \'\(key)\'")
        } else if key.contains("text_blocks") || key.contains("to_pred") {
          // Print keys related to the problem areas even if not renamed
          print("DEBUG: Key Processed (Unchanged?): \'\(key)\' (Original: \'\(originalKey)\')")
        }

        finalWeights[key] = value
      }
      return finalWeights
    }
    // --- End Helper ---

    // Attempt to load duration predictor
    var durationPredictor: DurationPredictor?
    let durationModelURL = modelDirectoryURL.appendingPathComponent("duration_v2.safetensors")
    do {
      var durationModelWeights = try loadArrays(url: durationModelURL)
      // Convert duration predictor weights
      durationModelWeights = convertWeights(durationModelWeights)

      let durationTransformer = DurationTransformer(
        dim: 512, depth: 8, heads: 8, dimHead: 64, ffMult: 2,
        textNumEmbeds: vocab.count, textDim: 512, convLayers: 2
      )
      let predictorMelSpec = MelSpec(filterbank: filterbank)
      // Assuming melSpec doesn't need @ModuleInfo as filterbank isn't a Module?
      // predictorMelSpec.freeze(recursive: false, keys: ["filterbank"]) // Freezing might need ModuleInfo? Check if needed.
      let predictor = DurationPredictor(
        transformer: durationTransformer,
        melSpec: predictorMelSpec,
        vocabCharMap: vocab
      )
      try predictor.update(
        parameters: ModuleParameters.unflattened(durationModelWeights), verify: [.all])
      durationPredictor = predictor
    } catch {
      print("Warning: no duration predictor model found: \(error)")
    }

    // Initialize the main model structure (always non-quantized initially)
    let dit = DiT(
      dim: 1024, depth: 22, heads: 16, ffMult: 2,
      textNumEmbeds: vocab.count, textDim: 512, convLayers: 4
    )
    let f5tts = F5TTS(
      transformer: dit,
      melSpec: MelSpec(filterbank: filterbank),  // Same question about ModuleInfo for melSpec here
      vocabCharMap: vocab,
      durationPredictor: durationPredictor
    )

    // Load weights or quantize structure THEN load weights
    var weightsToLoad: [String: MLXArray]

    if let bits = quantizationBits {
      // --- Quantize Swift Model Structure FIRST ---
      print("Quantizing model structure to \(bits)-bit...")
      let groupSize = 64
      quantize(
        model: f5tts,
        groupSize: groupSize,
        bits: bits,
        filter: { path, module in
          // Example: Quantize only Linear layers divisible by groupSize
          guard let linearLayer = module as? Linear else { return false }
          // Ensure weight exists and has at least 2 dimensions for shape check
          guard linearLayer.parameters().keys.contains("weight"), linearLayer.weight.ndim >= 2
          else { return false }
          return linearLayer.weight.shape[1] % groupSize == 0
        },
        apply: { module, groupSize, bits in
          // Custom apply logic: Convert Linear to QuantizedLinear
          if let linearLayer = module as? Linear {
            // Add check to ensure weight exists before creating QuantizedLinear
            guard linearLayer.parameters().keys.contains("weight") else { return nil }
            // Use the standard initializer instead of deprecated .from
            return QuantizedLinear(linearLayer, groupSize: groupSize, bits: bits)
          }
          return nil  // Return nil if module is not Linear or has no weight
        }
      )

      // --- Load Pre-Quantized Weights ---
      print("Loading pre-quantized weights from \(modelFilename)...")
      let originalQuantizedWeights = try loadArrays(url: modelURL)
      // --- Apply Weight Conversion to loaded weights ---
      weightsToLoad = convertWeights(originalQuantizedWeights)

    } else {
      // --- Load Non-Quantized Weights ---
      print("Loading non-quantized weights from \(modelFilename)...")
      let originalWeights = try loadArrays(url: modelURL)
      // --- Apply Weight Conversion ---
      weightsToLoad = convertWeights(originalWeights)
    }

    // Load the appropriately prepared weights
    print("Updating model with final weights...")
    try f5tts.update(parameters: ModuleParameters.unflattened(weightsToLoad), verify: [.all])

    // Evaluate parameters after loading
    print("Evaluating final parameters...")
    eval(f5tts.parameters())
    print("Model loaded successfully.")

    return f5tts
  }
}

// MARK: - Utilities

extension F5TTS {
  public static var sampleRate: Int = 24000
  public static var hopLength: Int = 256
  public static var framesPerSecond: Double = .init(sampleRate) / Double(hopLength)

  public static func loadAudioArray(url: URL) throws -> MLXArray {
    try AudioUtilities.loadAudioFile(url: url)
  }

  public static func referenceAudio() throws -> (MLXArray, String) {
    guard let url = Bundle.module.url(forResource: "test_en_1_ref_short", withExtension: "wav")
    else {
      throw F5TTSError.unableToLoadReferenceAudio
    }

    return try (
      self.loadAudioArray(url: url),
      "Some call me nature, others call me mother nature."
    )
  }

  public static func normalizeAudio(audio: MLXArray, targetRMS: Double = 0.1) -> MLXArray {
    let rms = Double(audio.square().mean().sqrt().item(Float.self))
    if rms < targetRMS {
      return audio * targetRMS / rms
    }
    return audio
  }

  public static func estimatedDuration(
    refAudio: MLXArray, refText: String, text: String, speed: Double = 1.0
  ) -> TimeInterval {
    let refDurationInFrames = refAudio.shape[0] / self.hopLength
    let refTextLength = refText.utf8.count
    let genTextLength = text.utf8.count

    let refAudioToTextRatio = Double(refDurationInFrames) / Double(refTextLength)
    let textLength = Double(genTextLength) / speed
    let estimatedDurationInFrames = Int(refAudioToTextRatio * textLength)

    let estimatedDuration = TimeInterval(estimatedDurationInFrames) / Self.framesPerSecond
    print(
      "Using duration of \(estimatedDuration) seconds (\(estimatedDurationInFrames) frames) for generated speech."
    )

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

  let padded: MLXArray
  switch ndim {
  case 1:
    padded = MLX.padded(t, widths: [.init((0, length - seqLen))], value: paddingValue)
  case 2:
    padded = MLX.padded(
      t, widths: [.init((0, 0)), .init((0, length - seqLen))], value: paddingValue)
  case 3:
    padded = MLX.padded(
      t, widths: [.init((0, 0)), .init((0, length - seqLen)), .init((0, 0))], value: paddingValue)
  default:
    fatalError("Unsupported padding dims: \(ndim)")
  }

  return padded[0..., .ellipsis]
}

func padSequence(_ t: [MLXArray], paddingValue: Float = 0) -> MLXArray {
  let maxLen = t.map { $0.shape.last ?? 0 }.max() ?? 0
  let t = MLX.stacked(t, axis: 0)
  return padToLength(t, length: maxLen, value: paddingValue)
}

func listStrToIdx(_ text: [String], vocabCharMap: [String: Int], paddingValue: Int = -1) -> MLXArray
{
  let listIdxTensors = text.map { str in str.map { char in vocabCharMap[String(char), default: 0] }
  }
  let mlxArrays = listIdxTensors.map { MLXArray($0) }
  let paddedText = padSequence(mlxArrays, paddingValue: Float(paddingValue))
  return paddedText.asType(.int32)
}
