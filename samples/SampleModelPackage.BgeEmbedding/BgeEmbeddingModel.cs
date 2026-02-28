using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.BgeEmbedding;

/// <summary>
/// BGE-small-en-v1.5 embedding model package.
/// For retrieval tasks, queries should be prefixed with "Represent this sentence: "
/// This prefix is baked into the model package — consumers don't need to know about it.
/// </summary>
public static class BgeEmbeddingModel
{
    private const string QueryPrefix = "Represent this sentence: ";

    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(BgeEmbeddingModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    /// <summary>
    /// Creates an IEmbeddingGenerator for BGE-small embeddings.
    /// The query prefix ("Represent this sentence: ") is automatically applied during embedding generation.
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

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        var transformer = estimator.Fit(dummyData);
        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
    }

    /// <summary>Prepends the BGE query prefix for retrieval tasks.</summary>
    public static string PrependQueryPrefix(string query) => QueryPrefix + query;

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class TextData { public string Text { get; set; } = ""; }
}
