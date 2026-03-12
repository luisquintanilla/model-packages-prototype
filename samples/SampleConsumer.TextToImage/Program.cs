using SampleModelPackage.TextToImage;
using System.Diagnostics;

Console.WriteLine("=== Text-to-Image Generation Demo ===\n");

Console.WriteLine("1. Creating image generator...");
Console.WriteLine("   (Model will be downloaded and cached on first run — ~4 GB)\n");

var sw = Stopwatch.StartNew();
var generator = await TextToImageModel.CreateGeneratorAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Generating image from prompt...");
var prompt = "a cat sitting on a beach at sunset";
Console.WriteLine($"   Prompt: \"{prompt}\"");

using var image = generator.Generate(prompt, seed: 42);
Console.WriteLine($"   Generated image: {image.Width}x{image.Height}");

Console.WriteLine("\n3. Model info:");
var info = await TextToImageModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

generator.Dispose();
Console.WriteLine("\nDone!");
