using Microsoft.ML.Data;
using SampleModelPackage.ImageSegmentation;
using System.Diagnostics;

Console.WriteLine("=== Image Segmentation Demo ===\n");

Console.WriteLine("1. Creating image segmenter...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var segmenter = await ImageSegmentationModel.CreateSegmenterAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Segmenter ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Segmenting image...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var mask = segmenter.Segment(image);

Console.WriteLine($"Mask size: {mask.Width}x{mask.Height}");

var uniqueClasses = mask.ClassIds.Distinct().OrderBy(id => id).ToArray();
Console.WriteLine($"Unique classes found: {uniqueClasses.Length}\n");

foreach (var classId in uniqueClasses)
{
    var label = mask.Labels is not null && classId < mask.Labels.Length
        ? mask.Labels[classId]
        : $"class_{classId}";
    var pixelCount = mask.ClassIds.Count(id => id == classId);
    var percentage = (float)pixelCount / mask.ClassIds.Length * 100;
    Console.WriteLine($"  [{classId}] {label}: {pixelCount} pixels ({percentage:F1}%)");
}

Console.WriteLine("\n3. Model info:");
var info = await ImageSegmentationModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

segmenter.Dispose();
Console.WriteLine("\nDone!");
