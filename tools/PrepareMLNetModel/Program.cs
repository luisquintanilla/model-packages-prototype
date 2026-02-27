using SampleModelPackage.MLNet;

// Paths
var onnxPath = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "ModelPackages", "ModelCache", "sentence-transformers", "all-MiniLM-L6-v2", "main", "model.onnx");

// Use absolute path — we know the repo root
var repoRoot = @"C:\Dev\model-packages-prototype";
var vocabPath = Path.Combine(repoRoot, "samples", "SampleModelPackage.Onnx", "vocab.txt");

var outputDir = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "ModelPackages", "PreparedModels", "sentence-transformers", "all-MiniLM-L6-v2");

Directory.CreateDirectory(outputDir);
var outputPath = Path.Combine(outputDir, "pipeline.mlnet");

Console.WriteLine($"ONNX model:  {onnxPath}");
Console.WriteLine($"Vocab:       {vocabPath}");
Console.WriteLine($"Output:      {outputPath}");

if (!File.Exists(onnxPath))
{
    Console.Error.WriteLine("ERROR: ONNX model not found. Run SampleConsumer.Onnx first to cache it.");
    return 1;
}

if (!File.Exists(vocabPath))
{
    Console.Error.WriteLine($"ERROR: vocab.txt not found at {vocabPath}");
    return 1;
}

Console.WriteLine("\nBuilding .mlnet pipeline...");
var (path, sha256) = await PrepareModel.BuildAndSaveAsync(onnxPath, vocabPath, outputPath);

var fileInfo = new FileInfo(path);
Console.WriteLine($"\nDone!");
Console.WriteLine($"  Path:   {path}");
Console.WriteLine($"  Size:   {fileInfo.Length}");
Console.WriteLine($"  SHA256: {sha256}");
Console.WriteLine($"\nUpdate model-manifest.json with these values.");

return 0;
