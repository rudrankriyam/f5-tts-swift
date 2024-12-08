import Foundation
import MLX
import MLXNN
import MLXRandom

class TextEmbedding: Module {
    let text_embed: Embedding
    var extraModeling: Bool = false
    var precomputeMaxPos: Int = 4096
    var freqsCis: MLXArray?
    var text_blocks: Sequential?

    init(textNumEmbeds: Int, textDim: Int, convLayers: Int = 0, convMult: Int = 2) {
        self.text_embed = Embedding(embeddingCount: textNumEmbeds + 1, dimensions: textDim)

        if convLayers > 0 {
            self.extraModeling = true
            self.freqsCis = precomputeFreqsCis(dim: textDim, end: precomputeMaxPos)
            self.text_blocks = Sequential(
                layers: (0 ..< convLayers).map { _ in ConvNeXtV2Block(dim: textDim, intermediateDim: textDim * convMult) }
            )
        }

        super.init()
    }

    func callAsFunction(_ inText: MLXArray, seqLen: Int, dropText: Bool = false) -> MLXArray {
        var text = inText + MLXArray([1])
        let batchSize = text.shape[0]
        let textLen = text.shape[1]

        if textLen > seqLen {
            text = text[0..., 0 ..< seqLen]
        }

        if textLen < seqLen {
            text = MLX.padded(text, widths: [.init((0, 0)), .init((0, seqLen - textLen))], value: MLXArray(0))
        }

        if dropText {
            text = MLX.zeros(like: text)
        }

        var output = text_embed(text)

        if extraModeling, let freqsCis = freqsCis, let textBlocks = text_blocks {
            let batchStart = MLX.zeros([batchSize], type: Int32.self)
            let posIdx = getPosEmbedIndices(start: batchStart, length: seqLen, maxPos: precomputeMaxPos)
            let textPosEmbed = freqsCis[posIdx]
            output = output + textPosEmbed
            output = textBlocks(output)
        }

        return output
    }
}

class InputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        let weight = MLXRandom.uniform(
            low: -sqrt(1.0 / Float(melDim * 2 + textDim)),
            high: sqrt(1.0 / Float(melDim * 2 + textDim)),
            [outDim, melDim * 2 + textDim]
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

// Transformer backbone using DiT blocks

public class DiT: Module {
    let dim: Int
    let depth: Int
    let time_embed: TimestepEmbedding
    let text_embed: TextEmbedding
    let input_embed: InputEmbedding
    let rotary_embed: RotaryEmbedding
    let transformer_blocks: [DiTBlock]
    let norm_out: AdaLayerNormZero_Final
    let proj_out: Linear

    init(
        dim: Int,
        depth: Int = 8,
        heads: Int = 8,
        dimHead: Int = 64,
        dropout: Float = 0.1,
        ffMult: Int = 4,
        melDim: Int = 100,
        textNumEmbeds: Int = 256,
        textDim: Int? = nil,
        convLayers: Int = 0
    ) {
        self.dim = dim
        let actualTextDim = textDim ?? melDim
        self.time_embed = TimestepEmbedding(dim: dim)
        self.text_embed = TextEmbedding(textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
        self.input_embed = InputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
        self.rotary_embed = RotaryEmbedding(dim: dimHead)
        self.depth = depth

        self.transformer_blocks = (0 ..< depth).map { _ in
            DiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout)
        }

        self.norm_out = AdaLayerNormZero_Final(dim: dim)
        
        let projWeight = MLXRandom.uniform(
            low: -sqrt(1.0 / Float(dim)),
            high: sqrt(1.0 / Float(dim)),
            [melDim, dim]
        )
        let projBias = MLXArray.zeros([melDim])
        self.proj_out = Linear(weight: projWeight, bias: projBias)

        super.init()
    }

    func callAsFunction(
        x: MLXArray,
        cond: MLXArray,
        text: MLXArray,
        time: MLXArray,
        dropAudioCond: Bool,
        dropText: Bool,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]

        let time = (time.ndim == 0) ? MLX.repeated(time.expandedDimensions(axis: 0), count: batchSize, axis: 0) : time
        let t = time_embed(time)
        let textEmbed = text_embed(text, seqLen: seqLen, dropText: dropText)
        var x = input_embed(cond: cond, textEmbed: textEmbed)

        let rope = rotary_embed.forwardFromSeqLen(seqLen)

        for block in transformer_blocks {
            x = block(x, t: t, mask: mask, rope: rope)
        }

        x = norm_out(x, emb: t)
        return proj_out(x)
    }
}
