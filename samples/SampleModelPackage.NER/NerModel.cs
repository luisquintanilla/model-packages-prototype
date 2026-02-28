using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.NER;

public static class NerModel
{
    private static readonly string[] Labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(NerModel).Assembly));

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    public static async Task<OnnxNerTransformer> CreateNerPipelineAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        var mlContext = new MLContext();
        var nerOptions = new OnnxNerOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            Labels = Labels,
            InputColumnName = "Text",
            OutputColumnName = "Entities",
            MaxTokenLength = 128,
            BatchSize = 8,
        };

        var estimator = mlContext.Transforms.OnnxNer(nerOptions);
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
