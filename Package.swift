// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "f5-tts-swift",
    platforms: [.macOS(.v14), .iOS(.v16), .visionOS(.v1)],
    products: [
        .library(
            name: "F5TTS",
            targets: ["F5TTS"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.13"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/lucasnewman/vocos-swift.git", from: "0.0.1")
    ],
    targets: [
        .target(
            name: "F5TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Vocos", package: "vocos-swift"),
            ],
            path: "Sources/F5TTS",
            resources: [
                .copy("Resources/test_en_1_ref_short.wav"),
                .copy("Resources/mel_filters.npy")
            ]
        ),
        .executableTarget(
            name: "f5-tts-generate",
            dependencies: [
                "F5TTS",
                .product(name: "Vocos", package: "vocos-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/f5-tts-generate"
        )
    ]
)
