using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextGeneration.OnnxGenAI;
using ModelPackages;

namespace SampleModelPackage.TextGeneration;

/// <summary>
/// Public API for the Phi-3-mini text generation model package.
/// Downloads the ONNX GenAI model on first use, caches it locally.
/// </summary>
public static class Phi3Model
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(Phi3Model).Assembly));

    /// <summary>Returns local path to the cached model directory.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an OnnxTextGenerationTransformer for local text generation.
    /// </summary>
    public static async Task<OnnxTextGenerationTransformer> CreateGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var modelPath = await EnsureModelAsync(options, ct);
        // The model path points to a directory containing the ONNX GenAI model files
        var modelDir = Path.GetDirectoryName(modelPath)!;

        var mlContext = new MLContext();
        var genOptions = new OnnxTextGenerationOptions
        {
            ModelPath = modelDir,
            MaxLength = 256,
            Temperature = 0.7f,
            TopP = 0.9f,
            SystemPrompt = "You are a helpful assistant. Be concise."
        };

        var estimator = mlContext.Transforms.OnnxTextGeneration(genOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        return estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class TextData
    {
        public string Text { get; set; } = "";
    }
}
