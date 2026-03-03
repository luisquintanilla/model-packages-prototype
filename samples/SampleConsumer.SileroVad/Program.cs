using MLNet.Audio.Core;
using SampleModelPackage.SileroVad;
using System.Diagnostics;

Console.WriteLine("=== Silero VAD Demo ===\n");

Console.WriteLine("1. Creating voice activity detector...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var vad = await SileroVadModel.CreateVadAsync(
    threshold: 0.5f,
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   VAD ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Detecting speech segments...");
var audio = AudioIO.LoadWav("test.wav");
var segments = vad.DetectSpeech(audio);

Console.WriteLine($"\nFound {segments.Length} speech segments:");
foreach (var seg in segments)
    Console.WriteLine($"  [{seg.Start:mm\\:ss\\.fff} → {seg.End:mm\\:ss\\.fff}] confidence: {seg.Confidence:F3}");

Console.WriteLine("\n3. Model info:");
var info = await SileroVadModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

vad.Dispose();
Console.WriteLine("\nDone!");
