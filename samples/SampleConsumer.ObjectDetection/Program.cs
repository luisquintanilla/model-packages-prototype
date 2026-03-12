using Microsoft.ML.Data;
using SampleModelPackage.ObjectDetection;
using System.Diagnostics;

Console.WriteLine("=== Object Detection Demo ===\n");

Console.WriteLine("1. Creating object detector...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var detector = await ObjectDetectionModel.CreateDetectorAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Detector ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Detecting objects...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var detections = detector.Detect(image);

Console.WriteLine($"Found {detections.Length} objects:");
foreach (var box in detections)
    Console.WriteLine($"  → {box}");

Console.WriteLine("\n3. Model info:");
var info = await ObjectDetectionModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

detector.Dispose();
Console.WriteLine("\nDone!");
