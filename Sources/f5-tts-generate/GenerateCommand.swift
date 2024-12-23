import ArgumentParser
import F5TTS
import Foundation
import MLX
import Vocos

@main
struct GenerateAudio: AsyncParsableCommand {
    @Argument(help: "Text to generate speech from")
    var text: String
    
    @Option(name: .long, help: "Duration of the generated audio in seconds")
    var duration: Double?
    
    @Option(name: .long, help: "Path to the reference audio file")
    var refAudioPath: String?
    
    @Option(name: .long, help: "Text spoken in the reference audio")
    var refAudioText: String?
    
    @Option(name: .long, help: "Model name to use")
    var model: String = "lucasnewman/f5-tts-mlx"
    
    @Option(name: .long, help: "Output path for the generated audio")
    var outputPath: String = "output.wav"
    
    @Option(name: .long, help: "Strength of classifier free guidance")
    var cfg: Double = 2.0
    
    @Option(name: .long, help: "Coefficient for sway sampling")
    var sway: Double = -1.0
    
    @Option(name: .long, help: "Speed factor for the duration heuristic")
    var speed: Double = 1.0
    
    @Option(name: .long, help: "Seed for noise generation")
    var seed: Int?
    
    func run() async throws {
        print("Loading F5-TTS model...")
        let f5tts = try await F5TTS.fromPretrained(repoId: model) { progress in
            print("  -- \(progress.completedUnitCount) of \(progress.totalUnitCount)")
        }
        
        let startTime = Date()
        
        let generatedAudio = try await f5tts.generate(
            text: text,
            referenceAudioURL: refAudioPath != nil ? URL(filePath: refAudioPath!) : nil,
            referenceAudioText: refAudioText,
            duration: duration,
            cfg: cfg,
            sway: sway,
            speed: speed,
            seed: seed
        )
        
        let elapsedTime = Date().timeIntervalSince(startTime)
        print("Generated \(Double(generatedAudio.shape[0]) / Double(F5TTS.sampleRate)) seconds of audio in \(elapsedTime) seconds.")
        
        try AudioUtilities.saveAudioFile(url: URL(filePath: outputPath), samples: generatedAudio)
        print("Saved audio to: \(outputPath)")
    }
}
