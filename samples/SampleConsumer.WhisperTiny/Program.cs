using Microsoft.ML;
using MLNet.Audio.Core;
using SampleModelPackage.WhisperTiny;
using System.Diagnostics;

Console.WriteLine("=== Whisper Tiny Speech-to-Text Demo ===\n");

Console.WriteLine("1. Creating speech-to-text model...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var mlContext = new MLContext();
var stt = await WhisperTinyModel.CreateSpeechToTextAsync(
    language: "en",
    mlContext: mlContext,
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Model ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Transcribing audio...");
var audio = AudioIO.LoadWav("test.wav");
var transcriptions = stt.Transcribe(new[] { audio });

for (int i = 0; i < transcriptions.Length; i++)
    Console.WriteLine($"   Transcription: {transcriptions[i]}");

Console.WriteLine("\n3. Model info:");
var info = await WhisperTinyModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

stt.Dispose();
Console.WriteLine("\nDone!");
