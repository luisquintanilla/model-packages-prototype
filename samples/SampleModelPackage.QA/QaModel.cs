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
        var modelPath = await EnsureModelAsync(options, ct);
        var vocabPath = ExtractEmbeddedVocab();

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

    private static string ExtractEmbeddedVocab()
    {
        var assembly = typeof(QaModel).Assembly;
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("vocab.txt", StringComparison.OrdinalIgnoreCase));

        if (resourceName == null)
            throw new FileNotFoundException("Embedded resource 'vocab.txt' not found in assembly.");

        var tempDir = Path.Combine(Path.GetTempPath(), "modelpackages-vocab-qa");
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

    private sealed class QaInput
    {
        public string Question { get; set; } = "";
        public string Context { get; set; } = "";
    }
}
