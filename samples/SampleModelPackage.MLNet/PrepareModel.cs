using System.Security.Cryptography;
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

namespace SampleModelPackage.MLNet;

/// <summary>
/// Helper for model package authors to prepare a .mlnet pipeline artifact.
/// Usage: Build the ML.NET pipeline from raw ONNX + vocab, save as .mlnet, get SHA256 for manifest.
/// </summary>
public static class PrepareModel
{
    /// <summary>
    /// Builds the ML.NET pipeline from raw ONNX model + vocab and saves as .mlnet file.
    /// Returns the path to the saved .mlnet file and its SHA256 hash.
    /// </summary>
    public static async Task<(string Path, string Sha256)> BuildAndSaveAsync(
        string onnxModelPath,
        string vocabPath,
        string outputPath)
    {
        var mlContext = new MLContext();
        var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
        {
            ModelPath = onnxModelPath,
            TokenizerPath = vocabPath,
            Pooling = PoolingStrategy.MeanPooling,
            Normalize = true,
            BatchSize = 32
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new TextData { Text = "" } });
        var transformer = estimator.Fit(dummyData);

        transformer.Save(outputPath);
        transformer.Dispose();

        var sha256 = await ComputeSha256Async(outputPath);
        return (outputPath, sha256);
    }

    private static async Task<string> ComputeSha256Async(string filePath)
    {
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hash = await sha.ComputeHashAsync(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private sealed class TextData
    {
        public string Text { get; set; } = "";
    }
}
