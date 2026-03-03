using Microsoft.ML;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.WhisperBase;

/// <summary>
/// Whisper Base speech-to-text model package.
/// 74M parameters — good balance of accuracy and speed.
/// Uses raw ONNX encoder + decoder from onnx-community.
/// </summary>
public static class WhisperBaseModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(WhisperBaseModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxWhisperTransformer> CreateSpeechToTextAsync(
        string language = "en",
        MLContext? mlContext = null,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        mlContext ??= new MLContext();

        var whisperOptions = new OnnxWhisperOptions
        {
            EncoderModelPath = files.GetPath("onnx/encoder_model.onnx"),
            DecoderModelPath = files.GetPath("onnx/decoder_model_merged.onnx"),
            Language = language,
            NumMelBins = 80,
            MaxTokens = 256,
            SampleRate = 16000
        };

        var estimator = new OnnxWhisperEstimator(mlContext, whisperOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        return (OnnxWhisperTransformer)estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
