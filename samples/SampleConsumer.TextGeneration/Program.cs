using SampleModelPackage.TextGeneration;
using System.Diagnostics;

Console.WriteLine("=== Model Package E2E Demo (Text Generation: Phi-3-mini Local) ===\n");

Console.WriteLine("1. Creating text generator...");
Console.WriteLine("   (Model will be downloaded and cached on first run — ~2.3 GB)\n");

var sw = Stopwatch.StartNew();
var transformer = await Phi3Model.CreateGeneratorAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   Generator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Generating text...");
var prompts = new[] { "What is machine learning?", "Explain .NET in one sentence." };
var responses = transformer.Generate(prompts);

for (int i = 0; i < prompts.Length; i++)
{
    Console.WriteLine($"   Prompt:   \"{prompts[i]}\"");
    Console.WriteLine($"   Response: \"{responses[i]}\"");
    Console.WriteLine();
}

Console.WriteLine("3. Model info:");
var info = await Phi3Model.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
