using MLNet.ImageGeneration.OnnxGenAI;
using ModelPackages;

namespace SampleModelPackage.TextToImage;

/// <summary>
/// Stable Diffusion v1.4 text-to-image generation model package.
/// Generates images from text prompts using ONNX GenAI runtime.
/// </summary>
public static class TextToImageModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(TextToImageModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxImageGenerationTransformer> CreateGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        // ModelDirectory needs to be the root containing text_encoder/, unet/, vae_decoder/
        var textEncoderDir = Path.GetDirectoryName(files.GetPath("text_encoder/model.onnx"))!;
        var modelDirectory = Path.GetDirectoryName(textEncoderDir)!;

        var genOptions = new OnnxImageGenerationOptions
        {
            ModelDirectory = modelDirectory,
            VocabPath = files.GetPath("tokenizer/vocab.json"),
            MergesPath = files.GetPath("tokenizer/merges.txt"),
            NumInferenceSteps = 20,
            GuidanceScale = 7.5f,
            Width = 512,
            Height = 512
        };

        return new OnnxImageGenerationTransformer(genOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
