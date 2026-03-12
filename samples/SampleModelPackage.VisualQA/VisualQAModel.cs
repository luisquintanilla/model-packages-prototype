using MLNet.Image.Core;
using Microsoft.Extensions.AI;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using ModelPackages;

namespace SampleModelPackage.VisualQA;

/// <summary>
/// GIT-Base (TextVQA) visual question answering model package.
/// Answers questions about image content.
/// Uses the same captioning architecture as ImageCaptioning but fine-tuned for VQA.
/// Also supports IChatClient via MEAI.
/// </summary>
public static class VisualQAModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(VisualQAModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageCaptioningTransformer> CreateTransformerAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var vqaOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GITVQA,
            MaxLength = 30
        };

        return new OnnxImageCaptioningTransformer(vqaOptions);
    }

    public static async Task<IChatClient> CreateChatClientAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var vqaOptions = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = files.GetPath("encoder.onnx"),
            DecoderModelPath = files.GetPath("decoder.onnx"),
            VocabPath = files.GetPath("vocab.txt"),
            PreprocessorConfig = PreprocessorConfig.GITVQA,
            MaxLength = 30
        };

        return new OnnxImageCaptioningChatClient(vqaOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
