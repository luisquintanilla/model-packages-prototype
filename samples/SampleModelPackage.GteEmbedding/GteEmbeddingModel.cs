using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.GteEmbedding;

/// <summary>
/// GTE-small embedding model package.
/// GTE models work well without any prefix — straightforward symmetric embeddings.
/// </summary>
public static class GteEmbeddingModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(GteEmbeddingModel).Assembly));

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

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class TextData { public string Text { get; set; } = ""; }
}
