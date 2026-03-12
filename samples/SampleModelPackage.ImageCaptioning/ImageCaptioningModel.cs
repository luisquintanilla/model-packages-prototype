using MLNet.Image.Core;
using Microsoft.Extensions.AI;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using ModelPackages;

namespace SampleModelPackage.ImageCaptioning;

/// <summary>
/// GIT-Base (COCO) image captioning model package.
/// Generates natural language captions for images.
/// Also supports IChatClient via MEAI for conversational image understanding.
/// </summary>
public static class ImageCaptioningModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ImageCaptioningModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageCaptioningTransformer> CreateCaptionerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var captioningOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };

        var estimator = new OnnxImageCaptioningEstimator(captioningOptions);
        return estimator.Fit(null!);
    }

    public static async Task<IChatClient> CreateChatClientAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var captioningOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };

        return new OnnxImageCaptioningChatClient(captioningOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
