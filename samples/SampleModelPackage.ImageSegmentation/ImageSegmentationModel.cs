using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Segmentation;
using ModelPackages;

namespace SampleModelPackage.ImageSegmentation;

/// <summary>
/// SegFormer-B0 image segmentation model package.
/// Segments images into semantic regions (ADE20K 150 classes).
/// </summary>
public static class ImageSegmentationModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ImageSegmentationModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageSegmentationTransformer> CreateSegmenterAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var segmentationOptions = new OnnxImageSegmentationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.SegFormer,
            ResizeToOriginal = true
        };

        var estimator = new OnnxImageSegmentationEstimator(segmentationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
