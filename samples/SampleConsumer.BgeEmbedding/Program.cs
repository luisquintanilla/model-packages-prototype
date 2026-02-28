using SampleModelPackage.BgeEmbedding;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Package E2E Demo (Embedding: BGE-small Asymmetric Retrieval) ===\n");

var sw = Stopwatch.StartNew();
var generator = await BgeEmbeddingModel.CreateEmbeddingGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

// BGE uses "Represent this sentence: " prefix for queries in retrieval
var query = "What is machine learning?";
var corpus = new[]
{
    "Machine learning is a subset of AI focused on learning from data.",
    "The weather in Paris is mild in spring.",
    "Deep learning uses neural networks with many layers.",
    "How to bake chocolate chip cookies at home.",
};

// Embed query with prefix, corpus without
var queryWithPrefix = BgeEmbeddingModel.PrependQueryPrefix(query);
var allTexts = new[] { queryWithPrefix }.Concat(corpus).ToArray();
var embeddings = await generator.GenerateAsync(allTexts);

Console.WriteLine($"Query: \"{query}\" (with BGE prefix)\n");
Console.WriteLine("Ranked results:");
var queryArr = embeddings[0].Vector.ToArray();
var ranked = corpus.Select((doc, i) =>
{
    float sim = TensorPrimitives.CosineSimilarity(queryArr.AsSpan(), embeddings[i + 1].Vector.Span);
    return (Similarity: sim, Document: doc);
}).OrderByDescending(x => x.Similarity).ToList();

foreach (var item in ranked)
{
    Console.WriteLine($"   [{item.Similarity:F4}] {item.Document}");
}

Console.WriteLine("\nDone!");
