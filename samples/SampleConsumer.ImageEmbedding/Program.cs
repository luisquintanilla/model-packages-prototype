using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using SampleModelPackage.ImageEmbedding;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Image Embedding Demo ===\n");

Console.WriteLine("1. Creating embedding generator...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var generator = await ImageEmbeddingModel.CreateEmbeddingGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Generating embeddings for images...");
using var image1 = MLImage.CreateFromFile("cat.jpg");
using var image2 = MLImage.CreateFromFile("dog.jpg");
using var image3 = MLImage.CreateFromFile("cat2.jpg");

var images = new[] { image1, image2, image3 };
var embeddings = await generator.GenerateAsync(images);

Console.WriteLine($"   Generated {embeddings.Count} embeddings of dim {embeddings[0].Vector.Length}\n");

Console.WriteLine("3. Cosine similarity (similar images should score higher):");
var e1 = embeddings[0].Vector.ToArray();
var e2 = embeddings[1].Vector.ToArray();
var e3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"   cat vs dog:   {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e2.AsSpan()):F4}");
Console.WriteLine($"   cat vs cat2:  {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e3.AsSpan()):F4}");
Console.WriteLine($"   dog vs cat2:  {TensorPrimitives.CosineSimilarity(e2.AsSpan(), e3.AsSpan()):F4}");

Console.WriteLine("\n4. Model info:");
var info = await ImageEmbeddingModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

Console.WriteLine("\nDone!");
