using SampleModelPackage.Onnx;
using Microsoft.Extensions.AI;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Package E2E Demo (Track 1: Raw ONNX from HuggingFace) ===\n");

// --- 1. Create embedding generator (auto-downloads model on first run) ---
Console.WriteLine("1. Creating embedding generator...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
IEmbeddingGenerator<string, Embedding<float>> generator =
    await MiniLMModel.CreateEmbeddingGeneratorAsync(new()
    {
        Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
    });
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

// --- 2. Generate embeddings ---
Console.WriteLine("2. Generating embeddings...");
var texts = new[]
{
    "What is machine learning?",
    "ML.NET is a machine learning framework for .NET",
    "How to cook pasta",
    "Deep learning and neural networks"
};

var embeddings = await generator.GenerateAsync(texts);
Console.WriteLine($"   Generated {embeddings.Count} embeddings, dimension: {embeddings[0].Vector.Length}\n");

// --- 3. Cosine similarity ---
Console.WriteLine("3. Cosine Similarity:");
for (int i = 0; i < texts.Length; i++)
{
    for (int j = i + 1; j < texts.Length; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(
            embeddings[i].Vector.Span, embeddings[j].Vector.Span);
        Console.WriteLine($"   \"{texts[i]}\" vs \"{texts[j]}\": {sim:F4}");
    }
}

// --- 4. Cached model access ---
Console.WriteLine("\n4. Cached model access:");
sw.Restart();
await MiniLMModel.EnsureModelAsync();
Console.WriteLine($"   Second call: {sw.ElapsedMilliseconds}ms (cached)\n");

// --- 5. Model info ---
Console.WriteLine("5. Model info:");
var info = await MiniLMModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

// --- 6. Override instructions ---
Console.WriteLine("\n6. To customize model source:");
Console.WriteLine("   Set MODELPACKAGES_SOURCE=company-mirror");
Console.WriteLine("   Or add model-sources.json next to your .csproj");

Console.WriteLine("\nDone!");
