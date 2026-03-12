using Microsoft.ML.Data;
using SampleModelPackage.ZeroShotClassification;
using System.Diagnostics;

Console.WriteLine("=== Zero-Shot Image Classification Demo ===\n");

Console.WriteLine("1. Creating zero-shot classifier...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var candidateLabels = new[]
{
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person"
};

var sw = Stopwatch.StartNew();
var classifier = await ZeroShotClassificationModel.CreateClassifierAsync(
    candidateLabels,
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Classifying image against custom labels...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var results = classifier.Classify(image);

Console.WriteLine("Predictions:");
foreach (var (label, probability) in results)
    Console.WriteLine($"  → {label}: {probability:P1}");

Console.WriteLine("\n3. Model info:");
var info = await ZeroShotClassificationModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

classifier.Dispose();
Console.WriteLine("\nDone!");
