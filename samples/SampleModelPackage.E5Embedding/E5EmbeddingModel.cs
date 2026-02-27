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
        var modelPath = await EnsureModelAsync(options, ct);
        var vocabPath = ExtractEmbeddedVocab();

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

    private static string ExtractEmbeddedVocab()
    {
        var assembly = typeof(E5EmbeddingModel).Assembly;
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("vocab.txt", StringComparison.OrdinalIgnoreCase));
        if (resourceName == null)
            throw new FileNotFoundException("Embedded resource 'vocab.txt' not found.");
        var tempDir = Path.Combine(Path.GetTempPath(), "modelpackages-vocab-e5");
        Directory.CreateDirectory(tempDir);
        var vocabPath = Path.Combine(tempDir, "vocab.txt");
        if (!File.Exists(vocabPath))
        {
            using var stream = assembly.GetManifestResourceStream(resourceName)!;
            using var file = File.Create(vocabPath);
            stream.CopyTo(file);
        }
        return vocabPath;
    }

    private sealed class TextData { public string Text { get; set; } = ""; }
}
