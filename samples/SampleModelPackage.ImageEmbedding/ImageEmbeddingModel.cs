using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.MEAI;
using ModelPackages;

namespace SampleModelPackage.ImageEmbedding;

/// <summary>
/// CLIP ViT-Base image embedding model package.
/// Generates 512-dim L2-normalized embeddings from images.
/// Implements IEmbeddingGenerator&lt;MLImage, Embedding&lt;float&gt;&gt; via MEAI.
/// </summary>
public static class ImageEmbeddingModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(ImageEmbeddingModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<IEmbeddingGenerator<MLImage, Embedding<float>>> CreateEmbeddingGeneratorAsync(
        ModelOptions? options = null, CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var embeddingOptions = new OnnxImageEmbeddingOptions
        {
            ModelPath = files.PrimaryModelPath,
            PreprocessorConfig = PreprocessorConfig.CLIP,
            Pooling = PoolingStrategy.ClsToken,
            Normalize = true
        };

        var transformer = new OnnxImageEmbeddingTransformer(embeddingOptions);
        return new OnnxImageEmbeddingGenerator(transformer);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
