using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.Reranking;

public static class RerankModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(RerankModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    public static async Task<OnnxRerankerTransformer> CreateRerankerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        var mlContext = new MLContext();
        var rerankOptions = new OnnxRerankerOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            QueryColumnName = "Query",
            DocumentColumnName = "Document",
            OutputColumnName = "Score",
            MaxTokenLength = 512,
            BatchSize = 8,
        };

        var estimator = new OnnxRerankerEstimator(mlContext, rerankOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new QueryDocument { Query = "", Document = "" } });
        return estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class QueryDocument
    {
        public string Query { get; set; } = "";
        public string Document { get; set; } = "";
    }
}
