using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.MLNet;

/// <summary>
/// Public API for the all-MiniLM-L6-v2 embedding model package.
/// Uses a pre-built .mlnet pipeline zip (fetched on demand via Core SDK).
/// </summary>
public static class MiniLMModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(MiniLMModel).Assembly));

    /// <summary>Returns local path to the cached .mlnet pipeline file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator by loading the pre-built .mlnet pipeline.
    /// No Fit() step needed — the pipeline is already built.
    /// </summary>
    public static async Task<IEmbeddingGenerator<string, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var mlnetPath = await EnsureModelAsync(options, ct);
        var mlContext = new MLContext();
        var transformer = OnnxTextEmbeddingTransformer.Load(mlContext, mlnetPath);
        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);
}
