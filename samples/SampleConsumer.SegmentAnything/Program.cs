using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.SegmentAnything;
using SampleModelPackage.SegmentAnything;
using System.Diagnostics;

Console.WriteLine("=== Segment Anything (SAM2) Demo ===\n");

Console.WriteLine("1. Creating SAM2 transformer...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await SegmentAnythingModel.CreateTransformerAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Transformer ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Segmenting with point prompt...");
using var image = MLImage.CreateFromFile("test-image.jpg");

var pointPrompt = SegmentAnythingPrompt.FromPoint(256f, 256f);
var pointResult = transformer.Segment(image, pointPrompt);

Console.WriteLine($"   Masks: {pointResult.NumMasks}");
Console.WriteLine($"   Best IoU: {pointResult.GetBestIoU():F4}");
Console.WriteLine($"   Mask pixels: {pointResult.GetBestMask().Count(v => v > 0)} foreground");

Console.WriteLine("\n3. Segmenting with bounding box prompt...");
var boxPrompt = SegmentAnythingPrompt.FromBoundingBox(128f, 128f, 384f, 384f);
var boxResult = transformer.Segment(image, boxPrompt);

Console.WriteLine($"   Masks: {boxResult.NumMasks}");
Console.WriteLine($"   Best IoU: {boxResult.GetBestIoU():F4}");
Console.WriteLine($"   Mask pixels: {boxResult.GetBestMask().Count(v => v > 0)} foreground");

Console.WriteLine("\n4. Cached embedding (multiple prompts)...");
var embedding = transformer.EncodeImage(image);
for (int i = 0; i < 3; i++)
{
    float x = 128f + i * 128f;
    var prompt = SegmentAnythingPrompt.FromPoint(x, 256f);
    var result = transformer.Segment(embedding, prompt);
    Console.WriteLine($"   Point ({x:F0}, 256) → IoU={result.GetBestIoU():F4}");
}

Console.WriteLine("\n5. Model info:");
var info = await SegmentAnythingModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
