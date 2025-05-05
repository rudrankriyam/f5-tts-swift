import Foundation
import MLX
import MLXFFT
import MLXFast
import MLXLinalg
import MLXNN
import MLXRandom

// rotary positional embedding related

class RotaryEmbedding: Module {
  let inv_freq: MLXArray
  let interpolationFactor: Float

  init(
    dim: Int,
    useXpos: Bool = false,
    scaleBase: Int = 512,
    interpolationFactor: Float = 1.0,
    base: Float = 10000.0,
    baseRescaleFactor: Float = 1.0
  ) {
    let adjustedBase = base * pow(baseRescaleFactor, Float(dim) / Float(dim - 2))
    self.inv_freq =
      1.0
      / pow(adjustedBase, MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim))

    assert(interpolationFactor >= 1.0, "Interpolation factor must be >= 1.0")
    self.interpolationFactor = interpolationFactor

    super.init()
  }

  func forwardFromSeqLen(_ seqLen: Int) -> (MLXArray, Float) {
    let t = MLXArray(0..<seqLen).asType(.float32)
    return callAsFunction(t)
  }

  func callAsFunction(_ t: MLXArray) -> (MLXArray, Float) {
    var freqs = MLX.matmul(
      t.expandedDimensions(axis: 1).asType(inv_freq.dtype), inv_freq.expandedDimensions(axis: 0))
    freqs = freqs / interpolationFactor

    freqs = MLX.stacked([freqs, freqs], axis: -1)
    let newShape = Array(
      freqs.shape.dropLast(2) + [
        freqs.shape[freqs.shape.count - 2] * freqs.shape[freqs.shape.count - 1]
      ]
    )
    freqs = MLX.reshaped(freqs, newShape)
    return (freqs, 1.0)
  }
}

func precomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0, thetaRescaleFactor: Float = 1.0)
  -> MLXArray
{
  let range = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)[0..<(dim / 2)]
  let freqs = 1.0 / MLX.pow(MLXArray(theta), range / Float(dim))

  let t = MLXArray(0..<end).asType(.float32)
  let outerFreqs = MLX.outer(t, freqs).asType(.float32)

  let freqsCos = outerFreqs.cos()
  let freqsSin = outerFreqs.sin()

  return MLX.concatenated([freqsCos, freqsSin], axis: -1)
}

func getPosEmbedIndices(start: MLXArray, length: Int, maxPos: Int, scale: Float = 1.0) -> MLXArray {
  let scaleArray = MLX.ones(like: start).asType(.float32) * scale

  let pos =
    MLX.expandedDimensions(start, axis: 1)
    + (MLXArray(0..<length).expandedDimensions(axis: 0) * scaleArray.expandedDimensions(axis: 1))
    .asType(.int32)

  return MLX.where(pos .< maxPos, pos, maxPos - 1)
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
  let shape = x.shape
  let newShape = Array(shape.dropLast() + [shape.last! / 2, 2])
  let reshapedX = x.reshaped(newShape)

  let x1x2 = reshapedX.split(parts: 2, axis: -1)
  let x1 = x1x2[0]
  let x2 = x1x2[1]

  let squeezedX1 = x1.squeezed(axis: -1)
  let squeezedX2 = x2.squeezed(axis: -1)

  let stackedX = MLX.stacked([-squeezedX2, squeezedX1], axis: -1)

  let finalShape = Array(
    stackedX.shape.dropLast(2) + [
      stackedX.shape[stackedX.shape.count - 2] * stackedX.shape[stackedX.shape.count - 1]
    ])
  let result = stackedX.reshaped(finalShape)

  return result
}

func applyRotaryPosEmb(t: MLXArray, freqs: MLXArray, scale: Float = 1.0) -> MLXArray {
  let rotDim = freqs.shape[freqs.shape.count - 1]
  let seqLen = t.shape[t.shape.count - 2]

  let freqsTrimmed = freqs[(-seqLen)..., 0...]
  let scaleAdjusted = MLXArray(scale)

  var freqsRearranged = freqsTrimmed
  if t.ndim == 4 && freqsRearranged.ndim == 3 {
    freqsRearranged = freqsRearranged.reshaped([
      freqsRearranged.shape[0], 1, freqsRearranged.shape[1], freqsRearranged.shape[2],
    ])
  }

  let tRotated = t[.ellipsis, 0..<rotDim]
  let tUnrotated = t[.ellipsis, rotDim..<t.shape[t.shape.count - 1]]
  let rotatedT =
    (tRotated * freqsRearranged.cos() * scaleAdjusted)
    + (rotateHalf(tRotated) * freqsRearranged.sin() * scaleAdjusted)
  let out = MLX.concatenated([rotatedT, tUnrotated], axis: -1)

  return out
}

// mel spec

public class MelSpec: Module {
  let sampleRate: Int
  let nFFT: Int
  let hopLength: Int
  let nMels: Int
  let filterbank: MLXArray

  init(
    sampleRate: Int = 24000,
    nFFT: Int = 1024,
    hopLength: Int = 256,
    nMels: Int = 100,
    filterbank: MLXArray
  ) {
    self.sampleRate = sampleRate
    self.nFFT = nFFT
    self.hopLength = hopLength
    self.nMels = nMels
    self.filterbank = filterbank
  }

  public func callAsFunction(x: MLXArray) -> MLXArray {
    logMelSpectrogram(
      audio: x, nMels: nMels, nFFT: nFFT, hopLength: hopLength, filterbank: filterbank)
  }

  public func stft(
    x: MLXArray, window: MLXArray, nperseg: Int, noverlap: Int? = nil, nfft: Int? = nil
  ) -> MLXArray {
    let nfft = nfft ?? nperseg
    let noverlap = noverlap ?? nfft
    let padding = nperseg / 2
    let x = MLX.padded(x, width: IntOrPair(padding))
    let strides = [noverlap, 1]
    let t = (x.shape[0] - nperseg + noverlap) / noverlap
    let shape = [t, nfft]
    let stridedX = MLX.asStrided(x, shape, strides: strides)
    return MLXFFT.rfft(stridedX * window)
  }

  public func logMelSpectrogram(
    audio: MLXArray, nMels: Int = 100, nFFT: Int = 1024, hopLength: Int = 256, filterbank: MLXArray
  ) -> MLXArray {
    let freqs = stft(x: audio, window: hanning(nFFT), nperseg: nFFT, noverlap: hopLength)
    let magnitudes = freqs[0..<freqs.shape[0] - 1].abs()
    let melSpec = MLX.matmul(magnitudes, filterbank.T)
    let logSpec = MLX.maximum(melSpec, 1e-5).log()
    return MLX.expandedDimensions(logSpec, axis: 0)
  }

  public func hanning(_ size: Int) -> MLXArray {
    let window = (0..<size).map { 0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(size - 1))) }
    return MLXArray(converting: window)
  }
}

// sinusoidal position embedding

class SinusPositionEmbedding: Module {
  let dim: Int

  init(dim: Int) {
    self.dim = dim
    super.init()
  }

  func callAsFunction(_ inputs: MLXArray) -> MLXArray {
    let scale: Float = 1000.0
    let halfDim = dim / 2

    let emb = log(10000.0) / Float(halfDim - 1)
    let expEmb = MLX.exp(MLXArray(0..<halfDim) * -emb)

    let expandedX = MLX.expandedDimensions(inputs, axis: 1)
    let expandedEmb = MLX.expandedDimensions(expEmb, axis: 0)

    let scaled = scale * expandedX * expandedEmb
    let sinEmb = MLX.sin(scaled)
    let cosEmb = MLX.cos(scaled)

    let output = MLX.concatenated([sinEmb, cosEmb], axis: -1)
    return output
  }
}

// convolutional position embedding

class ConvPositionEmbedding: Module {
  let conv1d: Sequential

  init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
    precondition(kernelSize % 2 != 0, "Kernel size must be odd.")

    self.conv1d = Sequential(layers: [
      GroupableConv1d(
        inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2,
        groups: groups),
      Mish(),
      GroupableConv1d(
        inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2,
        groups: groups),
      Mish(),
    ])

    super.init()
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    var input = x

    if let mask = mask {
      let expandedMask = MLX.expandedDimensions(mask, axis: -1)
      input = input * expandedMask
    }

    var output = conv1d(input)

    if let mask = mask {
      let expandedMask = MLX.expandedDimensions(mask, axis: -1)
      output = output * expandedMask
    }

    return output
  }
}

// global response normalization

class GRN: Module {
  var gamma: MLXArray
  var beta: MLXArray

  init(dim: Int) {
    self.gamma = MLX.zeros([1, 1, dim])
    self.beta = MLX.zeros([1, 1, dim])
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let Gx = MLXLinalg.norm(x, ord: 2, axis: 1, keepDims: true)
    let Nx = Gx / (Gx.mean(axis: -1, keepDims: true) + 1e-6)
    let output = gamma * (x * Nx) + beta + x
    return output
  }
}

// ConvNeXt-v2 block

open class GroupableConv1d: Module, UnaryLayer {
  public let weight: MLXArray
  public let bias: MLXArray?
  public let padding: Int
  public let dilation: Int
  public let groups: Int
  public let stride: Int

  convenience init(
    _ inputChannels: Int, _ outputChannels: Int, kernelSize: Int, padding: Int, dilation: Int,
    groups: Int
  ) {
    self.init(
      inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: kernelSize,
      padding: padding, dilation: dilation, groups: groups)
  }

  public init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true
  ) {
    let scale = sqrt(1 / Float(inputChannels * kernelSize))

    self.weight = uniform(
      low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups]
    )
    self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
  }

  open func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
    if let bias {
      y = y + bias
    }
    return y
  }
}

class ConvNeXtV2Block: Module, UnaryLayer {
  @ModuleInfo var dwconv: GroupableConv1d
  @ModuleInfo var norm: LayerNorm
  @ModuleInfo var pwconv1: Linear
  @ModuleInfo var act: GELU
  @ModuleInfo var grn: GRN
  @ModuleInfo var pwconv2: Linear

  init(dim: Int, intermediateDim: Int, dilation: Int = 1) {
    let padding = (dilation * (7 - 1)) / 2
    self.dwconv = GroupableConv1d(
      inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: padding, groups: dim)
    self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
    self.pwconv1 = Linear(dim, intermediateDim)
    self.act = GELU()
    self.grn = GRN(dim: intermediateDim)
    self.pwconv2 = Linear(intermediateDim, dim)

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let residual = x
    var out = dwconv(x)
    out = norm(out)
    out = pwconv1(out)
    out = act(out)
    out = grn(out)
    out = pwconv2(out)
    return residual + out
  }
}

// AdaLayerNormZero
// return with modulated x for attn input, and params for later mlp modulation

class AdaLayerNormZero: Module {
  let silu: SiLU
  let linear: Linear
  let norm: LayerNorm

  init(dim: Int) {
    self.silu = SiLU()
    self.linear = Linear(dim, dim * 6)
    self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, emb: MLXArray) -> (
    MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
  ) {
    let embProcessed = linear(silu(emb))
    let parts = embProcessed.split(parts: 6, axis: 1)
    let shiftMsa = parts[0]
    let scaleMsa = parts[1]
    let gateMsa = parts[2]
    let shiftMlp = parts[3]
    let scaleMlp = parts[4]
    let gateMlp = parts[5]

    let normX = norm(x)
    let modulatedX =
      normX * (MLXArray(1) + MLX.expandedDimensions(scaleMsa, axis: 1))
      + MLX.expandedDimensions(shiftMsa, axis: 1)
    return (modulatedX, gateMsa, shiftMlp, scaleMlp, gateMlp)
  }
}

// AdaLayerNormZero for final layer
// return only with modulated x for attn input, cuz no more mlp modulation

class AdaLayerNormZero_Final: Module {
  let silu: SiLU
  let linear: Linear
  let norm: LayerNorm

  init(dim: Int) {
    self.silu = SiLU()
    self.linear = Linear(dim, dim * 2)
    self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, emb: MLXArray? = nil) -> MLXArray {
    guard let emb = emb else {
      fatalError("Embedding tensor must not be nil")
    }

    let embProcessed = linear(silu(emb))

    let scaleAndShift = embProcessed.split(parts: 2, axis: 1)
    let scale = scaleAndShift[0]
    let shift = scaleAndShift[1]

    let modulatedX =
      norm(x) * (MLXArray(1) + scale.expandedDimensions(axis: 1))
      + shift.expandedDimensions(axis: 1)

    return modulatedX
  }
}

// feed forward

class FeedForward: Module {
  let ff: Sequential

  init(
    dim: Int, dimOut: Int? = nil, mult: Int = 4, dropout: Float = 0.0, approximate: String = "none"
  ) {
    let innerDim = Int(dim * mult)
    let outputDim = dimOut ?? dim

    let activation = GELU(approximation: approximate == "tanh" ? .tanh : .none)

    let projectIn = Sequential(layers: [
      Linear(dim, innerDim),
      activation,
    ])

    self.ff = Sequential(layers: [
      projectIn,
      Dropout(p: dropout),
      Linear(innerDim, outputDim),
    ])

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    return ff(x)
  }
}

// attention

class Attention: Module {
  let dim: Int
  let heads: Int
  let innerDim: Int
  let dropout: Float

  let to_q: Linear
  let to_k: Linear
  let to_v: Linear
  let to_out: Sequential

  init(dim: Int, heads: Int = 8, dimHead: Int = 64, dropout: Float = 0.0) {
    self.dim = dim
    self.heads = heads
    self.innerDim = heads * dimHead
    self.dropout = dropout

    self.to_q = Linear(dim, innerDim)
    self.to_k = Linear(dim, innerDim)
    self.to_v = Linear(dim, innerDim)

    self.to_out = Sequential(layers: [
      Linear(innerDim, dim),
      Dropout(p: dropout),
    ])

    super.init()
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil)
    -> MLXArray
  {
    let batch = x.shape[0]
    let seqLen = x.shape[1]

    var query = to_q(x)
    var key = to_k(x)
    var value = to_v(x)

    if let rope {
      let (freqs, xposScale) = rope
      let qXposScale = xposScale
      let kXposScale = pow(xposScale, -1.0)

      query = applyRotaryPosEmb(t: query, freqs: freqs, scale: qXposScale)
      key = applyRotaryPosEmb(t: key, freqs: freqs, scale: kXposScale)
    }

    query = rearrangeQuery(query, heads: heads)
    key = rearrangeQuery(key, heads: heads)
    value = rearrangeQuery(value, heads: heads)

    var attnMask: MLXArray? = nil
    if let mask = mask {
      let reshapedMask = mask.reshaped([mask.shape[0], 1, 1, mask.shape[1]])
      attnMask = MLX.repeated(reshapedMask, count: heads, axis: 1)
    }

    let scaleFactor = 1.0 / sqrt(Double(query.shape[query.shape.count - 1]))
    var output = MLXFast.scaledDotProductAttention(
      queries: query, keys: key, values: value, scale: Float(scaleFactor), mask: attnMask)

    output = output.transposed(axes: [0, 2, 1, 3]).reshaped([batch, seqLen, -1])
    output = to_out(output)

    if let mask = mask {
      let maskReshaped = mask.reshaped([batch, seqLen, 1])
      output = MLX.where(maskReshaped, MLX.logicalNot(maskReshaped), 0.0)
    }

    return output
  }

  private func rearrangeQuery(_ query: MLXArray, heads: Int) -> MLXArray {
    let batchSize = query.shape[0]
    let seqLength = query.shape[1]
    let headDim = query.shape[2] / heads
    return query.reshaped([batchSize, seqLength, heads, headDim]).transposed(axes: [0, 2, 1, 3])
  }
}

// DiT block

class DiTBlock: Module {
  let attn_norm: AdaLayerNormZero
  let attn: Attention
  let ff_norm: LayerNorm
  let ff: FeedForward

  init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
    self.attn_norm = AdaLayerNormZero(dim: dim)
    self.attn = Attention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
    self.ff_norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self.ff = FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh")

    super.init()
  }

  func callAsFunction(
    _ x: MLXArray, t: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil
  ) -> MLXArray {
    let (norm, gateMsa, shiftMlp, scaleMlp, gateMlp) = attn_norm(x, emb: t)
    let attnOutput = attn(norm, mask: mask, rope: rope)
    var output = x + gateMsa.expandedDimensions(axis: 1) * attnOutput
    let normedOutput =
      ff_norm(output) * (1 + scaleMlp.expandedDimensions(axis: 1))
      + shiftMlp.expandedDimensions(axis: 1)
    let ffOutput = ff(normedOutput)
    output = output + MLX.expandedDimensions(gateMlp, axis: 1) * ffOutput
    return output
  }
}

// time step conditioning embedding

class TimestepEmbedding: Module {
  let time_embed: SinusPositionEmbedding
  let time_mlp: Sequential

  init(dim: Int, freqEmbedDim: Int = 256) {
    self.time_embed = SinusPositionEmbedding(dim: freqEmbedDim)

    self.time_mlp = Sequential(
      layers: [
        Linear(freqEmbedDim, dim),
        SiLU(),
        Linear(dim, dim),
      ]
    )

    super.init()
  }

  func callAsFunction(_ timestep: MLXArray) -> MLXArray {
    let timeHidden = time_embed(timestep)
    let time = time_mlp(timeHidden)
    return time
  }
}
