using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.Classification;

public static class SentimentModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SentimentModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    public static async Task<OnnxTextClassificationTransformer> CreateClassifierAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        var mlContext = new MLContext();
        var estimatorOptions = new OnnxTextClassificationOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            InputColumnName = "Text",
            Labels = ["NEGATIVE", "POSITIVE"],
            MaxTokenLength = 128,
            BatchSize = 8,
        };

        var estimator = new OnnxTextClassificationEstimator(mlContext, estimatorOptions);
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
