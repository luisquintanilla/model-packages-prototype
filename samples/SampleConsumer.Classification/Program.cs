using SampleModelPackage.Classification;
using System.Diagnostics;

Console.WriteLine("=== Model Package E2E Demo (Classification: Sentiment Analysis) ===\n");

Console.WriteLine("1. Creating sentiment classifier...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await SentimentModel.CreateClassifierAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Classifying text...");
var texts = new[]
{
    "I absolutely loved this movie! The acting was superb.",
    "This was the worst experience of my life.",
    "The food was okay, nothing special.",
    "What an amazing concert! Best night ever!",
    "I'm really disappointed with the quality.",
};

var results = transformer.Classify(texts.ToList());

for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"   \"{texts[i]}\"");
    Console.WriteLine($"     → {results[i].PredictedLabel} (confidence: {results[i].Confidence:P1})");
}

Console.WriteLine("\n3. Model info:");
var info = await SentimentModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
