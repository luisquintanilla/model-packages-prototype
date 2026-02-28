using SampleModelPackage.QA;
using System.Diagnostics;

Console.WriteLine("=== Model Package E2E Demo (QA: Extractive Question Answering) ===\n");

Console.WriteLine("1. Creating QA pipeline...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var transformer = await QaModel.CreateQaPipelineAsync(new()
{
    Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}")
});
Console.WriteLine($"\n   QA pipeline ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Answering questions...");

var questions = new[]
{
    "When was the Eiffel Tower built?",
    "What programming language is Python named after?",
    "What color is the sky on Venus?",
};
var contexts = new[]
{
    "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was constructed from 1887 to 1889.",
    "Python is a high-level programming language. It was named after the BBC comedy show Monty Python's Flying Circus.",
    "Venus is the second planet from the Sun. It is the hottest planet in our solar system.",
};

var answers = transformer.Answer(questions.ToList(), contexts.ToList());

for (int i = 0; i < questions.Length; i++)
{
    Console.WriteLine($"   Q: \"{questions[i]}\"");
    Console.WriteLine($"   Context: \"{contexts[i]}\"");
    if (answers[i].Answer.Length > 0)
        Console.WriteLine($"   Answer: \"{answers[i].Answer}\" (score: {answers[i].Score:F4})");
    else
        Console.WriteLine($"   Answer: <unanswerable> (score: {answers[i].Score:F4})");
    Console.WriteLine();
}

Console.WriteLine("3. Model info:");
var info = await QaModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

transformer.Dispose();
Console.WriteLine("\nDone!");
