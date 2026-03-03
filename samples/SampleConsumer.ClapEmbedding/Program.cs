using Microsoft.Extensions.AI;
using MLNet.Audio.Core;
using SampleModelPackage.ClapEmbedding;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== CLAP Audio Embedding Demo ===\n");

Console.WriteLine("1. Creating embedding generator...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var generator = await ClapEmbeddingModel.CreateEmbeddingGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Generating embeddings for audio files...");
var audio1 = AudioIO.LoadWav("dog_bark.wav");
var audio2 = AudioIO.LoadWav("cat_meow.wav");
var audio3 = AudioIO.LoadWav("dog_bark2.wav");

var audioInputs = new[] { audio1, audio2, audio3 };
var embeddings = await generator.GenerateAsync(audioInputs);

Console.WriteLine($"   Generated {embeddings.Count} embeddings of dim {embeddings[0].Vector.Length}\n");

Console.WriteLine("3. Cosine similarity (similar sounds should score higher):");
var e1 = embeddings[0].Vector.ToArray();
var e2 = embeddings[1].Vector.ToArray();
var e3 = embeddings[2].Vector.ToArray();

Console.WriteLine($"   dog_bark vs cat_meow:  {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e2.AsSpan()):F4}");
Console.WriteLine($"   dog_bark vs dog_bark2: {TensorPrimitives.CosineSimilarity(e1.AsSpan(), e3.AsSpan()):F4}");
Console.WriteLine($"   cat_meow vs dog_bark2: {TensorPrimitives.CosineSimilarity(e2.AsSpan(), e3.AsSpan()):F4}");

Console.WriteLine("\n4. Model info:");
var info = await ClapEmbeddingModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

Console.WriteLine("\nDone!");
