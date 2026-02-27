using SampleModelPackage.Reranking;
using System.Diagnostics;

Console.WriteLine("=== Model Package E2E Demo (Reranking: Cross-Encoder) ===\n");

Console.WriteLine("1. Creating reranker...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await RerankModel.CreateRerankerAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Reranker ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Reranking documents...");

var query = "What is machine learning?";
var documents = new[]
{
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The weather forecast predicts rain tomorrow in Seattle.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "How to bake chocolate chip cookies at home.",
    "ML.NET is a cross-platform machine learning framework for .NET developers.",
};

Console.WriteLine($"   Query: \"{query}\"\n");

var inputData = documents.Select(doc => new { Query = query, Document = doc }).ToArray();
var mlContext = new Microsoft.ML.MLContext();
var dataView = mlContext.Data.LoadFromEnumerable(inputData);
var results = transformer.Transform(dataView);
var scores = mlContext.Data.CreateEnumerable<ScoreResult>(results, reuseRowObject: false).ToList();

var ranked = scores
    .Select((s, i) => (Score: s.Score, Document: documents[i]))
    .OrderByDescending(x => x.Score)
    .ToList();

foreach (var (score, document) in ranked)
{
    Console.WriteLine($"   [{score:F4}] {document}");
}

Console.WriteLine("\n3. Model info:");
var info = await RerankModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");

public class ScoreResult
{
    public float Score { get; set; }
}
