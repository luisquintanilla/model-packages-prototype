using SampleModelPackage.NER;
using System.Diagnostics;

Console.WriteLine("=== Model Package E2E Demo (NER: Named Entity Recognition) ===\n");

Console.WriteLine("1. Creating NER pipeline...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await NerModel.CreateNerPipelineAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   NER pipeline ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Extracting entities...");
var texts = new[]
{
    "John Smith works at Microsoft in Seattle.",
    "Angela Merkel met with Emmanuel Macron in Berlin.",
    "The United Nations headquarters is in New York.",
};

var entities = transformer.ExtractEntities(texts.ToList());

for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"   Text: \"{texts[i]}\"");
    foreach (var e in entities[i])
    {
        Console.WriteLine($"     {e.EntityType}: \"{e.Word}\" [{e.StartChar}..{e.EndChar}] (score: {e.Score:F4})");
    }
    Console.WriteLine();
}

Console.WriteLine("3. Model info:");
var info = await NerModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
