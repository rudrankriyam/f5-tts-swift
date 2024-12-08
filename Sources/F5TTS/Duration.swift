import Foundation
import MLX
import MLXNN
import MLXRandom

class DurationInputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        let weight = MLXRandom.uniform(
            low: -sqrt(1.0 / Float(melDim + textDim)),
            high: sqrt(1.0 / Float(melDim + textDim)),
            [outDim, melDim + textDim]
        )
        let bias = MLXArray.zeros([outDim])
        self.proj = Linear(weight: weight, bias: bias)
        self.conv_pos_embed = ConvPositionEmbedding(dim: outDim)
        super.init()
    }

    func callAsFunction(
        cond: MLXArray,
        textEmbed: MLXArray
    ) -> MLXArray {
        let combined = MLX.concatenated([cond, textEmbed], axis: -1)
        var output = proj(combined)
        output = conv_pos_embed(output)
        return output
    }
}

public class DurationBlock: Module {
    let attn_norm: LayerNorm
    let attn: Attention
    let ff_norm: LayerNorm
    let ff: FeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
        self.attn_norm = LayerNorm(dimensions: dim)
        self.attn = Attention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
        self.ff_norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.ff = FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh")

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil) -> MLXArray {
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
    let text_embed: TextEmbedding
    let input_embed: DurationInputEmbedding
    let rotary_embed: RotaryEmbedding
    let transformer_blocks: [DurationBlock]
    let norm_out: RMSNorm
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
        self.text_embed = TextEmbedding(textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
        self.input_embed = DurationInputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
        self.rotary_embed = RotaryEmbedding(dim: dimHead)
        self.depth = depth

        self.transformer_blocks = (0 ..< depth).map { _ in
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

public class DurationPredictor: Module {
    enum DurationPredictorError: Error {
        case unableToLoadModel
        case unableToLoadReferenceAudio
        case unableToDetermineDuration
    }

    public let melSpec: MelSpec
    public let transformer: DurationTransformer

    let dim: Int
    let numChannels: Int
    let vocabCharMap: [String: Int]
    let to_pred: Sequential

    init(
        transformer: DurationTransformer,
        melSpec: MelSpec,
        vocabCharMap: [String: Int]
    ) {
        self.transformer = transformer
        self.melSpec = melSpec
        self.vocabCharMap = vocabCharMap
        self.dim = transformer.dim
        self.numChannels = melSpec.nMels
        
        let weight = MLXRandom.uniform(
            low: -sqrt(1.0 / Float(transformer.dim)),
            high: sqrt(1.0 / Float(transformer.dim)),
            [1, transformer.dim]
        )
        self.to_pred = Sequential(layers: [
            Linear(weight: weight, bias: nil),
            Softplus()
        ])
        
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
        output = to_pred(output).mean().reshaped([batch, -1])
        output.eval()

        return output
    }
}
