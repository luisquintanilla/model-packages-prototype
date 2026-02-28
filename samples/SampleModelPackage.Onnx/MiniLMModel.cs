using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.Onnx;

/// <summary>
/// Public API for the all-MiniLM-L6-v2 embedding model package.
/// Uses raw ONNX from HuggingFace (fetched on demand via Core SDK).
/// </summary>
public static class MiniLMModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(MiniLMModel).Assembly));

    /// <summary>Returns local path to the cached ONNX model file.</summary>
    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator backed by the local ONNX model.
    /// Downloads the model on first call, cached thereafter.
    /// Uses MLNet.TextInference.Onnx to build the ML.NET pipeline and wrap it as IEmbeddingGenerator.
    /// </summary>
    public static async Task<IEmbeddingGenerator<string, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        var mlContext = new MLContext();
        var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            Pooling = PoolingStrategy.MeanPooling,
            Normalize = true,
            BatchSize = 32
        });

        // Fit with dummy data (probes ONNX metadata, not training)
        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        var transformer = estimator.Fit(dummyData);

        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    public static Task VerifyModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.VerifyModelAsync(options, ct);

    private sealed class TextData
    {
        public string Text { get; set; } = "";
    }
}
