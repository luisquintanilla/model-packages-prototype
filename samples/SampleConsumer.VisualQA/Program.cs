using Microsoft.ML.Data;
using SampleModelPackage.VisualQA;
using System.Diagnostics;

Console.WriteLine("=== Visual Question Answering Demo ===\n");

Console.WriteLine("1. Creating VQA transformer...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await VisualQAModel.CreateTransformerAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Transformer ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Asking questions about image...");
using var image = MLImage.CreateFromFile("test-image.jpg");

var questions = new[]
{
    "What is shown in this image?",
    "How many objects are there?",
    "What color is the main object?"
};

foreach (var question in questions)
{
    var answer = transformer.AnswerQuestion(image, question);
    Console.WriteLine($"   Q: {question}");
    Console.WriteLine($"   A: {answer}\n");
}

Console.WriteLine("3. Generating caption...");
var caption = transformer.GenerateCaption(image);
Console.WriteLine($"   Caption: {caption}");

Console.WriteLine("\n4. Model info:");
var info = await VisualQAModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
