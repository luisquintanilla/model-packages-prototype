using Microsoft.ML;
using MLNet.Audio.Core;
using SampleModelPackage.AstAudioSet;
using System.Diagnostics;

Console.WriteLine("=== AST AudioSet Classification Demo ===\n");

Console.WriteLine("1. Creating audio classifier...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var mlContext = new MLContext();
var classifier = await AstAudioSetModel.CreateClassifierAsync(
    mlContext: mlContext,
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Classifier ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Classifying audio...");
var audio = AudioIO.LoadWav("test.wav");
var results = classifier.Classify(new[] { audio });

Console.WriteLine("Top predictions:");
foreach (var result in results)
    Console.WriteLine($"  → {result}");

Console.WriteLine("\n3. Model info:");
var info = await AstAudioSetModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

classifier.Dispose();
Console.WriteLine("\nDone!");
