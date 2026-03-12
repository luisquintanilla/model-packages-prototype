using Microsoft.ML.Data;
using SampleModelPackage.ImageCaptioning;
using System.Diagnostics;

Console.WriteLine("=== Image Captioning Demo ===\n");

Console.WriteLine("1. Creating image captioner...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var captioner = await ImageCaptioningModel.CreateCaptionerAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Captioner ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Generating caption...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var caption = captioner.GenerateCaption(image);
Console.WriteLine($"   Caption: {caption}");

Console.WriteLine("\n3. Visual question answering...");
var answer = captioner.AnswerQuestion(image, "What color is the sky?");
Console.WriteLine($"   Q: What color is the sky?");
Console.WriteLine($"   A: {answer}");

Console.WriteLine("\n4. Model info:");
var info = await ImageCaptioningModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

captioner.Dispose();
Console.WriteLine("\nDone!");
