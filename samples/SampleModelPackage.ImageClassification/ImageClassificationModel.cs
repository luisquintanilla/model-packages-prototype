using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Classification;
using ModelPackages;

namespace SampleModelPackage.ImageClassification;

/// <summary>
/// ViT-Base image classification model package.
/// Classifies images into 1000 ImageNet categories.
/// </summary>
public static class ImageClassificationModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ImageClassificationModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageClassificationTransformer> CreateClassifierAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var classificationOptions = new OnnxImageClassificationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var estimator = new OnnxImageClassificationEstimator(classificationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
