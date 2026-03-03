using Microsoft.ML;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.SileroVad;

public static class SileroVadModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SileroVadModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxVadTransformer> CreateVadAsync(
        float threshold = 0.5f,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var mlContext = new MLContext();

        var vadOptions = new OnnxVadOptions
        {
            ModelPath = files.PrimaryModelPath,
            Threshold = threshold,
            MinSpeechDuration = TimeSpan.FromMilliseconds(250),
            MinSilenceDuration = TimeSpan.FromMilliseconds(100),
            SpeechPad = TimeSpan.FromMilliseconds(30),
            WindowSize = 512,
            SampleRate = 16000
        };

        var estimator = mlContext.Transforms.OnnxVad(vadOptions);
        var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
        return (OnnxVadTransformer)estimator.Fit(emptyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class AudioInput { public float[] Audio { get; set; } = []; }
}
