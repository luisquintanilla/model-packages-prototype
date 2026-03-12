using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ZeroShot;
using ModelPackages;

namespace SampleModelPackage.ZeroShotClassification;

/// <summary>
/// CLIP ViT-Base zero-shot image classification model package.
/// Classifies images against arbitrary text labels without task-specific training.
/// Requires separate vision/text ONNX models and tokenizer files.
/// </summary>
public static class ZeroShotClassificationModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ZeroShotClassificationModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxZeroShotImageClassificationTransformer> CreateClassifierAsync(
        string[] candidateLabels,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var classificationOptions = new OnnxZeroShotImageClassificationOptions
        {
            ImageModelPath = files.GetPath("onnx/vision_model.onnx"),
            TextModelPath = files.GetPath("onnx/text_model.onnx"),
            VocabPath = files.GetPath("vocab.json"),
            MergesPath = files.GetPath("merges.txt"),
            CandidateLabels = candidateLabels,
            PreprocessorConfig = PreprocessorConfig.CLIP
        };

        var estimator = new OnnxZeroShotImageClassificationEstimator(classificationOptions);
        return estimator.Fit(null!);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
