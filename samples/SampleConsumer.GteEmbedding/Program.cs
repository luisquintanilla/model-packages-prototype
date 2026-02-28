using SampleModelPackage.GteEmbedding;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Package E2E Demo (Embedding: GTE-small Semantic Search) ===\n");

var sw = Stopwatch.StartNew();
var generator = await GteEmbeddingModel.CreateEmbeddingGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

// GTE works without any prefix — simple symmetric embeddings
var queries = new[] { "What is machine learning?", "How do I make coffee?" };
var corpus = new[]
{
    "Machine learning is a branch of AI that learns from data.",
    "The weather in Tokyo is hot and humid in summer.",
    "Neural networks are inspired by biological brain structures.",
    "How to make a perfect cup of coffee: grind fresh beans and use hot water.",
    "Supervised learning uses labeled training data.",
};

Console.WriteLine("Embedding corpus...");
var corpusEmbeddings = await generator.GenerateAsync(corpus);

foreach (var query in queries)
{
    Console.WriteLine($"\nQuery: \"{query}\"");
    var queryEmbedding = await generator.GenerateAsync(new[] { query });
    var queryArr = queryEmbedding[0].Vector.ToArray();

    var ranked = corpus.Select((doc, i) =>
    {
        float sim = TensorPrimitives.CosineSimilarity(queryArr.AsSpan(), corpusEmbeddings[i].Vector.Span);
        return (Similarity: sim, Document: doc);
    }).OrderByDescending(x => x.Similarity).ToList();

    Console.WriteLine("Top results:");
    foreach (var item in ranked.Take(3))
    {
        Console.WriteLine($"   [{item.Similarity:F4}] {item.Document}");
    }
}

Console.WriteLine("\nDone!");
