using Microsoft.ML.Data;
using SampleModelPackage.DepthEstimation;
using System.Diagnostics;

Console.WriteLine("=== Depth Estimation Demo ===\n");

Console.WriteLine("1. Creating depth estimator...");
Console.WriteLine("   (Model will be downloaded and cached on first run)\n");

var sw = Stopwatch.StartNew();
var estimator = await DepthEstimationModel.CreateEstimatorAsync(
    options: new() { Logger = msg => Console.WriteLine($"   [{sw.Elapsed:mm\\:ss}] {msg}") });
Console.WriteLine($"\n   Estimator ready in {sw.ElapsedMilliseconds}ms\n");

Console.WriteLine("2. Estimating depth...");
using var image = MLImage.CreateFromFile("test-image.jpg");
var depthMap = estimator.Estimate(image);

Console.WriteLine($"Depth map: {depthMap.Width}x{depthMap.Height}");
Console.WriteLine($"Raw depth range: [{depthMap.MinDepth:F2}, {depthMap.MaxDepth:F2}]");

var values = depthMap.Values;
Console.WriteLine($"\nNormalized depth stats:");
Console.WriteLine($"  Mean:   {values.Average():F4}");
Console.WriteLine($"  Min:    {values.Min():F4}");
Console.WriteLine($"  Max:    {values.Max():F4}");

Console.WriteLine("\n3. Model info:");
var info = await DepthEstimationModel.GetModelInfoAsync();
Console.WriteLine($"   Model ID:  {info.ModelId}");
Console.WriteLine($"   Source:    {info.ResolvedSource}");
Console.WriteLine($"   Cached at: {info.LocalPath}");

estimator.Dispose();
Console.WriteLine("\nDone!");
