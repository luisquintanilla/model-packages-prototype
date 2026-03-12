using Microsoft.ML.Data;
using SampleModelPackage.ImageClassification;
using System.Diagnostics;

Console.WriteLine("=== Image Classification Demo ===\n");

Console.WriteLine("1. Creating image classifier...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var classifier = await ImageClassificationModel.CreateClassifierAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Classifying image...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var results = classifier.Classify(image);

Console.WriteLine("Top predictions:");
foreach (var (label, probability) in results)
    Console.WriteLine($"  → {label}: {probability:P1}");

Console.WriteLine("\n3. Model info:");
var info = await ImageClassificationModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

classifier.Dispose();
Console.WriteLine("\nDone!");
