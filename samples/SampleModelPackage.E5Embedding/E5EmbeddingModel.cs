using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.E5Embedding;

/// <summary>
/// E5-small-v2 embedding model package.
/// E5 uses dual-prefix: "query: " for queries, "passage: " for documents.
/// These prefixes are baked into the model package.
/// </summary>
public static class E5EmbeddingModel
{
    private const string QueryPrefix = "query: ";
    private const string PassagePrefix = "passage: ";

    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(E5EmbeddingModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

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

    /// <summary>Prepends "query: " for E5 retrieval queries.</summary>
    public static string PrependQueryPrefix(string query) => QueryPrefix + query;

    /// <summary>Prepends "passage: " for E5 document passages.</summary>
    public static string PrependPassagePrefix(string passage) => PassagePrefix + passage;

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class TextData { public string Text { get; set; } = ""; }
}
