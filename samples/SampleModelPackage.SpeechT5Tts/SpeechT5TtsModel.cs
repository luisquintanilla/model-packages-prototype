using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;
using ModelPackages;

namespace SampleModelPackage.SpeechT5Tts;

/// <summary>
/// SpeechT5 text-to-speech model package.
/// Requires 5 files: encoder, decoder, vocoder, tokenizer, and speaker embedding.
/// ITextToSpeechClient is a prototype interface (not yet in MEAI).
/// </summary>
public static class SpeechT5TtsModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(SpeechT5TtsModel).Assembly));

    public static Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureFilesAsync(options, ct);

    public static async Task<OnnxSpeechT5TtsTransformer> CreateTtsAsync(
        MLContext? mlContext = null,
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        mlContext ??= new MLContext();

        var ttsOptions = new OnnxSpeechT5Options
        {
            EncoderModelPath = files.GetPath("encoder_model.onnx"),
            DecoderModelPath = files.GetPath("decoder_model_merged.onnx"),
            VocoderModelPath = files.GetPath("decoder_postnet_and_vocoder.onnx"),
            MaxMelFrames = 500,
            StopThreshold = 0.5f
        };

        return new OnnxSpeechT5TtsTransformer(mlContext, ttsOptions);
    }

    public static async Task<ITextToSpeechClient> CreateTtsClientAsync(
        ModelOptions? options = null,
        CancellationToken ct = default)
    {
        var files = await Package.Value.EnsureFilesAsync(options, ct);

        var ttsOptions = new OnnxSpeechT5Options
        {
            EncoderModelPath = files.GetPath("encoder_model.onnx"),
            DecoderModelPath = files.GetPath("decoder_model_merged.onnx"),
            VocoderModelPath = files.GetPath("decoder_postnet_and_vocoder.onnx"),
            MaxMelFrames = 500,
            StopThreshold = 0.5f
        };

        return new OnnxTextToSpeechClient(ttsOptions);
    }

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);
}
