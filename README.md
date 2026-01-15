
# F5 TTS for Swift
[![Star History Chart](https://api.star-history.com/svg?repos=rudrankriyam/f5-tts-swift&type=Date)](https://star-history.com/#rudrankriyam/f5-tts-swift&Date)


Implementation of [F5-TTS](https://arxiv.org/abs/2410.06885) in Swift, using the [MLX Swift](https://github.com/ml-explore/mlx-swift) framework.

You can listen to a [sample here](https://s3.amazonaws.com/lucasnewman.datasets/f5tts/sample.wav) that was generated in ~11 seconds on an M3 Max MacBook Pro.

See the [Python repository](https://github.com/lucasnewman/f5-tts-mlx) for additional details on the model architecture.

This repository is based on the original Pytorch implementation available [here](https://github.com/SWivid/F5-TTS).


## Installation

The `F5TTS` Swift package can be built and run from Xcode or SwiftPM.

A pretrained model is available [on Huggingface](https://hf.co/lucasnewman/f5-tts-mlx).


## Usage

```swift
import F5TTS

let f5tts = try await F5TTS.fromPretrained(repoId: "lucasnewman/f5-tts-mlx")

let generatedAudio = try await f5tts.generate(text: "The quick brown fox jumped over the lazy dog.")
```

The result is an MLXArray with 24kHz audio samples.

If you want to use your own reference audio sample, make sure it's a mono, 24kHz wav file of around 5-10 seconds:

```swift
let generatedAudio = try await f5tts.generate(
    text: "The quick brown fox jumped over the lazy dog.",
    referenceAudioURL: ...,
    referenceAudioText: "This is the caption for the reference audio."
)
```

You can convert an audio file to the correct format with ffmpeg like this:

```bash
ffmpeg -i /path/to/audio.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 /path/to/output_audio.wav
```

## Appreciation

[Yushen Chen](https://github.com/SWivid) for the original Pytorch implementation of F5 TTS and pretrained model.

[Phil Wang](https://github.com/lucidrains) for the E2 TTS implementation that this model is based on.

## Citations

```bibtex
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```

```bibtex
@inproceedings{Eskimez2024E2TE,
    title   = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author  = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270738197}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
