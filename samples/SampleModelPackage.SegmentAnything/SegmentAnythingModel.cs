using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.SegmentAnything;
using ModelPackages;

namespace SampleModelPackage.SegmentAnything;

/// <summary>
/// SAM2 Hiera-Tiny segment anything model package.
/// Segments any object in an image given point or box prompts.
/// Supports cached image embeddings for multi-prompt segmentation.
/// </summary>
public static class SegmentAnythingModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SegmentAnythingModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxSegmentAnythingTransformer> CreateTransformerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var samOptions = new OnnxSegmentAnythingOptions
        {
            EncoderModelPath = files.GetPath("sam2_hiera_tiny_encoder.onnx"),
            DecoderModelPath = files.GetPath("sam2_hiera_tiny_decoder.onnx"),
            PreprocessorConfig = PreprocessorConfig.SAM2
        };

        return new OnnxSegmentAnythingTransformer(samOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
