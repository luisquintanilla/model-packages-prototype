using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.AstAudioSet;

/// <summary>
/// Audio Spectrogram Transformer (AST) finetuned on AudioSet.
/// Classifies audio into 527 sound categories.
/// Uses 128 mel bins at 16kHz.
/// </summary>
public static class AstAudioSetModel
{
    // 527 AudioSet labels — sourced from config.json id2label mapping
    // https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/config.json
    // TODO: Download and embed the full 527 label array from config.json
    private static readonly string[] AudioSetLabels = GetAudioSetLabels();

    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(AstAudioSetModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxAudioClassificationTransformer> CreateClassifierAsync(
        MLContext? mlContext = null,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        mlContext ??= new MLContext();

        var classificationOptions = new OnnxAudioClassificationOptions
        {
            ModelPath = files.PrimaryModelPath,
            FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
            {
                NumMelBins = 128,
                FftSize = 400,
                HopLength = 160
            },
            Labels = AudioSetLabels,
            SampleRate = 16000
        };

        var pipeline = mlContext.Transforms.OnnxAudioClassification(classificationOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        return (OnnxAudioClassificationTransformer)pipeline.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private static string[] GetAudioSetLabels()
    {
        // First 10 labels shown for reference — full 527 labels must be sourced from:
        // https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/raw/main/config.json
        // Parse the "id2label" mapping, ordered by integer key 0..526
        return [
            "Speech", "Male speech, man speaking", "Female speech, woman speaking",
            "Child speech, kid speaking", "Conversation", "Narration, monologue",
            "Babbling", "Speech synthesizer", "Shout", "Bellow"
            // TODO: Add remaining 517 labels from config.json id2label mapping
        ];
    }

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
