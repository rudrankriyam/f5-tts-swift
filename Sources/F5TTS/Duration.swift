import Foundation
import MLX
import MLXNN

class DurationInputEmbedding: Module {
  @ModuleInfo var proj: Linear
  @ModuleInfo var conv_pos_embed: ConvPositionEmbedding

  init(melDim: Int, textDim: Int, outDim: Int) {
    self.proj = Linear(melDim + textDim, outDim)
    self.conv_pos_embed = ConvPositionEmbedding(dim: outDim)
    super.init()
  }

  func callAsFunction(
    cond: MLXArray,
    textEmbed: MLXArray
  ) -> MLXArray {
    var output = proj(MLX.concatenated([cond, textEmbed], axis: -1))
    output = conv_pos_embed(output) + output
    return output
  }
}

public class DurationBlock: Module {
  @ModuleInfo var attn_norm: LayerNorm
  @ModuleInfo var attn: Attention
  @ModuleInfo var ff_norm: LayerNorm
  @ModuleInfo var ff: FeedForward

  init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
    self.attn_norm = LayerNorm(dimensions: dim)
    self.attn = Attention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
    self.ff_norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self.ff = FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh")

    super.init()
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil)
    -> MLXArray
  {
    let norm = attn_norm(x)
    let attnOutput = attn(norm, mask: mask, rope: rope)
    var output = x + attnOutput
    let normedOutput = ff_norm(output)
    let ffOutput = ff(normedOutput)
    output = output + ffOutput
    return output
  }
}

public class DurationTransformer: Module {
  let dim: Int
  @ModuleInfo var text_embed: TextEmbedding
  @ModuleInfo var input_embed: DurationInputEmbedding
  @ModuleInfo var rotary_embed: RotaryEmbedding
  @ModuleInfo var transformer_blocks: [DurationBlock]
  @ModuleInfo var norm_out: RMSNorm
  let depth: Int

  init(
    dim: Int,
    depth: Int = 8,
    heads: Int = 8,
    dimHead: Int = 64,
    dropout: Float = 0.0,
    ffMult: Int = 4,
    melDim: Int = 100,
    textNumEmbeds: Int = 256,
    textDim: Int? = nil,
    convLayers: Int = 0
  ) {
    self.dim = dim
    let actualTextDim = textDim ?? melDim
    self.text_embed = TextEmbedding(
      textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
    self.input_embed = DurationInputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
    self.rotary_embed = RotaryEmbedding(dim: dimHead)
    self.depth = depth

    self.transformer_blocks = (0..<depth).map { _ in
      DurationBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout)
    }

    self.norm_out = RMSNorm(dimensions: dim)

    super.init()
  }

  func callAsFunction(
    cond: MLXArray,
    text: MLXArray,
    mask: MLXArray? = nil
  ) -> MLXArray {
    let seqLen = cond.shape[1]

    let textEmbed = text_embed(text, seqLen: seqLen)
    var x = input_embed(cond: cond, textEmbed: textEmbed)

    let rope = rotary_embed.forwardFromSeqLen(seqLen)

    for block in transformer_blocks {
      x = block(x, mask: mask, rope: rope)
    }

    return norm_out(x)
  }
}

// Custom Module to encapsulate the final layers of DurationPredictor
class PredictorHead: Module {
  @ModuleInfo var linear: Linear
  @ModuleInfo var activation: Softplus  // Softplus typically has no weights/params

  init(dim: Int) {
    self.linear = Linear(dim, 1, bias: false)
    self.activation = Softplus()
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = linear(x)
    y = activation(y)
    return y
  }
}

public class DurationPredictor: Module {
  enum DurationPredictorError: Error {
    case unableToLoadModel
    case unableToLoadReferenceAudio
    case unableToDetermineDuration
  }

  @ModuleInfo public var melSpec: MelSpec
  @ModuleInfo public var transformer: DurationTransformer

  let dim: Int
  let numChannels: Int
  let vocabCharMap: [String: Int]
  @ModuleInfo var to_pred: PredictorHead

  init(
    transformer: DurationTransformer,
    melSpec: MelSpec,
    vocabCharMap: [String: Int]
  ) {
    self.melSpec = melSpec
    self.numChannels = melSpec.nMels
    self.transformer = transformer
    self.dim = transformer.dim
    self.vocabCharMap = vocabCharMap

    self.to_pred = PredictorHead(dim: dim)

    super.init()
  }

  func callAsFunction(_ cond: MLXArray, text: [String]) -> MLXArray {
    var cond = cond

    // raw wave

    if cond.ndim == 2 {
      cond = cond.reshaped([cond.shape[1]])
      cond = melSpec(x: cond)
    }

    let batch = cond.shape[0]
    let condSeqLen = cond.shape[1]
    var lens = MLX.full([batch], values: condSeqLen, type: Int.self)

    // text

    let inputText = listStrToIdx(text, vocabCharMap: vocabCharMap)
    let textLens = (inputText .!= -1).sum(axis: -1)
    lens = MLX.maximum(textLens, lens)

    var output = transformer(cond: cond, text: inputText)

    // Use the new to_pred module
    output = to_pred(output)

    output = output.mean().reshaped([batch, -1])
    output.eval()

    return output
  }
}

// Helper function (assuming it's needed and correct)
func listStrToIdx(_ texts: [String], vocabCharMap: [String: Int]) -> MLXArray {
  // This function needs to return an MLXArray of indices
  // based on the input strings and vocabulary map.
  let batchSize = texts.count
  let maxLen = texts.map { $0.count }.max() ?? 0

  var flatIndices = [Int]()
  for text in texts {
    var indices = text.map { vocabCharMap[String($0)] ?? -1 }  // Assuming -1 for unknown/padding
    // Pad to maxLen
    indices += Array(repeating: -1, count: maxLen - indices.count)
    flatIndices.append(contentsOf: indices)
  }

  // Create MLXArray from flat array and reshape
  let array = MLXArray(flatIndices)
  // Reshape to [batchSize, maxLen]
  return array.reshaped([batchSize, maxLen])
}
