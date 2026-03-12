using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Detection;
using ModelPackages;

namespace SampleModelPackage.ObjectDetection;

/// <summary>
/// YOLOv8s object detection model package.
/// Detects objects in images with bounding boxes and labels.
/// </summary>
public static class ObjectDetectionModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ObjectDetectionModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxObjectDetectionTransformer> CreateDetectorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var detectionOptions = new OnnxObjectDetectionOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.5f,
            IouThreshold = 0.45f
        };

        var estimator = new OnnxObjectDetectionEstimator(detectionOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
