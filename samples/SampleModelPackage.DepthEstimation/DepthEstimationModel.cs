using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.DepthEstimation;
using ModelPackages;

namespace SampleModelPackage.DepthEstimation;

/// <summary>
/// DPT-Hybrid (MiDaS) depth estimation model package.
/// Estimates relative depth from monocular images.
/// </summary>
public static class DepthEstimationModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(DepthEstimationModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageDepthEstimationTransformer> CreateEstimatorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var depthOptions = new OnnxImageDepthEstimationOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.DPT,
            ResizeToOriginal = true
        };

        var estimator = new OnnxImageDepthEstimationEstimator(depthOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
