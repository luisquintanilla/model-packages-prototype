using SampleModelPackage.E5Embedding;
using System.Diagnostics;
using System.Numerics.Tensors;

Console.WriteLine("=== Model Package E2E Demo (Embedding: E5-small Dual-Prefix Retrieval) ===\n");

var sw = Stopwatch.StartNew();
var generator = await E5EmbeddingModel.CreateEmbeddingGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

// E5 uses "query: " for queries and "passage: " for documents
var query = "What is machine learning?";
var corpus = new[]
{
    "Machine learning is a branch of AI that learns from data.",
    "The weather in Tokyo is hot and humid in summer.",
    "Neural networks are inspired by biological brain structures.",
    "How to make a perfect cup of coffee.",
};

// Embed with E5 prefixes
var queryText = E5EmbeddingModel.PrependQueryPrefix(query);
var passageTexts = corpus.Select(E5EmbeddingModel.PrependPassagePrefix).ToArray();
var allTexts = new[] { queryText }.Concat(passageTexts).ToArray();
var embeddings = await generator.GenerateAsync(allTexts);

Console.WriteLine($"Query: \"{query}\" (with E5 \"query: \" prefix)\n");
Console.WriteLine("Ranked results (passages with \"passage: \" prefix):");
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
