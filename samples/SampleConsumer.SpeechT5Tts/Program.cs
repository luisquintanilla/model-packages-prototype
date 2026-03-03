using MLNet.Audio.Core;
using SampleModelPackage.SpeechT5Tts;
using System.Diagnostics;

Console.WriteLine("=== SpeechT5 Text-to-Speech Demo ===\n");

Console.WriteLine("1. Creating TTS client...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var tts = await SpeechT5TtsModel.CreateTtsClientAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   TTS ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Synthesizing speech...");
var response = await tts.GetAudioAsync("Hello, this is a test of the SpeechT5 text to speech model.");
Console.WriteLine($"   Duration: {response.Duration}");
Console.WriteLine($"   Voice: {response.Voice}");

Console.WriteLine("\n3. Saving to WAV...");
AudioIO.SaveWav("output.wav", response.Audio);
Console.WriteLine("   Saved to output.wav");

Console.WriteLine("\n4. Model info:");
var info = await SpeechT5TtsModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

Console.WriteLine("\nDone!");
