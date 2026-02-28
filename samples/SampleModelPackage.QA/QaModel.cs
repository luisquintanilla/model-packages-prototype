using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.QA;

public static class QaModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(QaModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    public static async Task<OnnxQaTransformer> CreateQaPipelineAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        var mlContext = new MLContext();
        var qaOptions = new OnnxQaOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            MaxTokenLength = 384,
            MaxAnswerLength = 30,
            BatchSize = 8,
        };

        var estimator = mlContext.Transforms.OnnxQa(qaOptions);
        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new QaInput { Question = "", Context = "" } });
        return estimator.Fit(dummyData);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class QaInput
    {
        public string Question { get; set; } = "";
        public string Context { get; set; } = "";
    }
}
