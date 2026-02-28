# Authoring Guide

This guide walks through creating a model package — a small NuGet package that lets consumers use an AI model with a single method call, while the large model binary is fetched and cached on demand.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- A model you want to package (ONNX format for this guide)
- The model's SHA256 hash and file size

## Concepts

A model package has three components:

1. **Manifest** (`model-manifest.json`) — Describes the model: where to download it, expected SHA256 hash, file size. Supports multiple files (e.g., ONNX model + vocabulary).
2. **Public API** — A static class that wires everything together and exposes a simple interface to consumers
3. **Small assets** (optional) — Files small enough to embed (config files, label maps). Tokenizer vocabularies are typically listed in the manifest and downloaded on demand.

The heavy model binary is *never* in the package. It's fetched at runtime.

## Step-by-Step: Track 1 (Raw ONNX from HuggingFace)

This track packages a raw ONNX model that lives on HuggingFace. The consumer's machine downloads the ONNX file and builds an ML.NET pipeline at runtime.

### 1. Create the project

```bash
dotnet new classlib -n MyModelPackage -f net10.0
cd MyModelPackage
dotnet add reference path/to/ModelPackages.csproj
dotnet add package MLNet.TextInference.Onnx --version 0.1.0-preview.1
```

### 2. Create the manifest

Create `model-manifest.json` in your project root. List all files the model needs — both the ONNX model and any tokenizer files:

```json
{
  "model": {
    "id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "main",
    "files": [
      {
        "path": "onnx/model.onnx",
        "sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "size": 90405214
      },
      {
        "path": "vocab.txt",
        "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        "size": 231508
      }
    ]
  },
  "sources": {
    "huggingface": {
      "type": "huggingface"
    }
  },
  "defaultSource": "huggingface"
}
```

The first file in the `files` array is the **primary model file** (returned by `EnsureModelAsync()`). Additional files (tokenizer vocabularies, config files) are downloaded alongside it when using `EnsureFilesAsync()`.

**Getting the SHA256:** For HuggingFace models, the SHA256 is the Git LFS OID. You can find it on the model page under "Files and versions" → click the file → look for the LFS pointer, or run:

```bash
# Using the HuggingFace CLI
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 onnx/model.onnx
sha256sum ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/blobs/<hash>
```

### 3. Embed the manifest

In your `.csproj`:

```xml
<ItemGroup>
  <EmbeddedResource Include="model-manifest.json" />
</ItemGroup>
```

> **Note:** Tokenizer files (vocab.txt, etc.) should be listed in the manifest `files` array instead of embedded as assembly resources. This keeps NuGet packages small and lets the SDK handle downloading, caching, and integrity verification for all files uniformly.

### 4. Create the public API

```csharp
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;
using ModelPackages;

namespace MyModelPackage;

public static class MyModel
{
    private static readonly Lazy<ModelPackage> Package = new(() =>
        ModelPackage.FromManifestResource(typeof(MyModel).Assembly));

    public static async Task<IEmbeddingGenerator<string, Embedding<float>>>
        CreateEmbeddingGeneratorAsync(ModelOptions? options = null, CancellationToken ct = default)
    {
        // 1. Ensure all files (ONNX model + vocab) are downloaded and cached
        var files = await Package.Value.EnsureFilesAsync(options, ct);
        var modelPath = files.PrimaryModelPath;
        var vocabPath = files.GetPath("vocab.txt");

        // 2. Build the ML.NET pipeline
        var mlContext = new MLContext();
        var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
        {
            ModelPath = modelPath,
            TokenizerPath = vocabPath,
            Pooling = PoolingStrategy.MeanPooling,
            Normalize = true,
            BatchSize = 32
        });

        var dummyData = mlContext.Data.LoadFromEnumerable(
            new[] { new TextInput { Text = "" } });
        var transformer = estimator.Fit(dummyData);

        return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
    }

    public static Task<string> EnsureModelAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.EnsureModelAsync(options, ct);

    public static Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null, CancellationToken ct = default)
        => Package.Value.GetModelInfoAsync(options, ct);

    private sealed class TextInput
    {
        public string Text { get; set; } = "";
    }
}
```

### 5. Add MSBuild integration (optional but recommended)

Create `build/MyModelPackage.props`:

```xml
<Project>
  <PropertyGroup>
    <ModelPackagesEnabled>true</ModelPackagesEnabled>
  </PropertyGroup>
</Project>
```

And include it in your `.csproj`:

```xml
<ItemGroup>
  <None Include="build\MyModelPackage.props" Pack="true" PackagePath="build\" />
</ItemGroup>
```

### 6. Pack and publish

```bash
dotnet pack -c Release
dotnet nuget push bin/Release/MyModelPackage.1.0.0.nupkg --source nuget.org
```

Your consumers can now do:

```bash
dotnet add package MyModelPackage
```

```csharp
var generator = await MyModel.CreateEmbeddingGeneratorAsync();
var embeddings = await generator.GenerateAsync(new[] { "Hello world" });
```

## Step-by-Step: Track 2 (Pre-built .mlnet Pipeline)

This track packages a pre-built `.mlnet` pipeline. The model author builds the ML.NET pipeline once and hosts the output file. The consumer just downloads and loads it — no `Fit()` step needed.

### 1. Build the .mlnet pipeline

You need the raw ONNX model and vocab file locally. Use the `PrepareModel` helper:

```csharp
var (path, sha256) = await PrepareModel.BuildAndSaveAsync(
    onnxModelPath: "/path/to/model.onnx",
    vocabPath: "/path/to/vocab.txt",
    outputPath: "/path/to/output/pipeline.mlnet"
);

Console.WriteLine($"SHA256: {sha256}");
Console.WriteLine($"Size: {new FileInfo(path).Length}");
```

Or use the included tool:

```bash
dotnet run --project tools/PrepareMLNetModel
```

### 2. Host the .mlnet file

Upload `pipeline.mlnet` to wherever you want consumers to download from:
- A CDN or file server
- Azure Blob Storage
- A local network share (using `file://` URL)

### 3. Create the manifest

```json
{
  "model": {
    "id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "main",
    "files": [
      {
        "path": "pipeline.mlnet",
        "sha256": "<sha256-from-step-1>",
        "size": <size-from-step-1>
      }
    ]
  },
  "sources": {
    "default": {
      "type": "direct",
      "url": "https://your-cdn.com/models/pipeline.mlnet"
    }
  },
  "defaultSource": "default"
}
```

### 4. Create the public API

The Track 2 API is simpler — no `Fit()`, no vocab extraction:

```csharp
public static async Task<IEmbeddingGenerator<string, Embedding<float>>>
    CreateEmbeddingGeneratorAsync(ModelOptions? options = null, CancellationToken ct = default)
{
    var mlnetPath = await Package.Value.EnsureModelAsync(options, ct);
    var mlContext = new MLContext();
    var transformer = OnnxTextEmbeddingTransformer.Load(mlContext, mlnetPath);
    return new OnnxEmbeddingGenerator(mlContext, transformer, ownsTransformer: true);
}
```

### 5. Pack and publish

Same as Track 1 — `dotnet pack` and `dotnet nuget push`.

## Consumer Experience

For both tracks, the consumer's code is nearly identical:

```csharp
// Track 1
using SampleModelPackage.Onnx;
var generator = await MiniLMModel.CreateEmbeddingGeneratorAsync();

// Track 2
using SampleModelPackage.MLNet;
var generator = await MiniLMModel.CreateEmbeddingGeneratorAsync();
```

The consumer doesn't know or care which track was used. They get `IEmbeddingGenerator<string, Embedding<float>>` and generate embeddings.

## Customizing Model Sources

Consumers can override where the model is downloaded from without changing code:

### Environment variable
```bash
export MODELPACKAGES_SOURCE=company-mirror
dotnet run
```

### Project-level config
Create `model-sources.json` next to your `.csproj`:
```json
{
  "sources": {
    "company-mirror": {
      "type": "mirror",
      "endpoint": "https://models.internal.company.com"
    }
  },
  "defaultSource": "company-mirror"
}
```

### Programmatic override
```csharp
var generator = await MiniLMModel.CreateEmbeddingGeneratorAsync(new ModelOptions
{
    Source = "file:///C:/local-models/model.onnx"
});
```

## Important Notes

### Multi-File Manifests

The manifest `files` array supports multiple entries. Use this when your model needs additional files beyond the ONNX binary (tokenizer vocabulary, config files, etc.):

```json
"files": [
  { "path": "onnx/model.onnx", "sha256": "...", "size": 133093490 },
  { "path": "vocab.txt", "sha256": "...", "size": 231508 }
]
```

In your facade, use `EnsureFilesAsync()` to download all files at once:

```csharp
var files = await Package.Value.EnsureFilesAsync(options, ct);
var modelPath = files.PrimaryModelPath;        // First file in manifest
var vocabPath = files.GetPath("vocab.txt");     // Any file by manifest path
```

The `ModelFiles` result type provides:
- `PrimaryModelPath` — path to the first file in the manifest
- `GetPath(manifestPath)` — path to any file by its manifest path
- `HasFile(manifestPath)` — check if a file is in the manifest
- `ModelDirectory` — directory containing the primary model file

### Extracting Embedded Resources

For resources that must be embedded in the NuGet package itself (e.g., custom label maps not available from HuggingFace), use the `ExtractResources()` utility instead of writing your own extraction boilerplate:

```csharp
var resourceDir = ModelPackage.ExtractResources(
    typeof(MyModel).Assembly, "MyModel");
var labelMapPath = Path.Combine(resourceDir, "labels.txt");
```

This extracts matching resources to a model-specific cache directory. Default patterns cover common tokenizer files (`vocab.txt`, `vocab.json`, `merges.txt`, `spm.model`, `tokenizer.json`, etc.). You can pass custom patterns:

```csharp
var resourceDir = ModelPackage.ExtractResources(
    typeof(MyModel).Assembly, "MyModel",
    filePatterns: ["labels.txt", "config.json"]);
```

> **Prefer multi-file manifests** over embedded resources for files available from HuggingFace. Embedded resources increase NuGet package size. Use `ExtractResources()` only for files that must ship with the package.

### OnnxRuntime Native Binaries

The inference library (`MLNet.TextInference.Onnx`) uses `Microsoft.ML.OnnxRuntime.Managed` — the managed-only wrapper. It does **not** include native OnnxRuntime binaries.

The **consumer application** (or the model package) must include:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.24.2" />
```

This ensures the correct native binaries for the consumer's platform (Windows x64, Linux x64, macOS arm64, etc.) are included.

### Embedded Resource Naming

When you embed `model-manifest.json` as an assembly resource, .NET prefixes the resource name with the project's default namespace. The Core SDK's `FromManifestResource` method handles this automatically by searching for a resource name ending with the expected filename.

### SHA256 for HuggingFace LFS Files

HuggingFace stores large files using Git LFS. The SHA256 in the manifest should be the **LFS OID** (the hash of the actual file content), not the hash of the LFS pointer. You can verify this by downloading the file and computing `sha256sum` locally.

## Task-Specific Patterns

The model-package pattern works across all AI inference tasks. Here's how the public API facade differs by task type:

### Embeddings

Return `IEmbeddingGenerator<string, Embedding<float>>` from `CreateEmbeddingGeneratorAsync()`. Different embedding models may require query/passage prefixes — bake these into the package so consumers don't need to know about them.

See: `SampleModelPackage.Onnx`, `SampleModelPackage.BgeEmbedding`, `SampleModelPackage.E5Embedding`, `SampleModelPackage.GteEmbedding`

### Classification

Expose a `CreateClassifierAsync()` method that returns an `OnnxTextClassificationTransformer`. The label list (e.g., `["NEGATIVE", "POSITIVE"]`) is defined in code as part of the estimator options. Consumers call classification methods on the returned transformer.

See: `SampleModelPackage.Classification`

### Named Entity Recognition

Expose a `CreateNerPipelineAsync()` method that returns an `OnnxNerTransformer`. Consumers call `ExtractEntities` on the transformer to get entity spans (text, label, score). The BIO label list is defined in code (e.g., as a `private static readonly string[] Labels` field).

See: `SampleModelPackage.NER`

### Question Answering

Return answer span + score from `AnswerAsync(question, context)`. The facade handles tokenizing the question-context pair.

See: `SampleModelPackage.QA`

### Reranking

Return scored + ranked documents from `RerankAsync(query, documents)`. The facade handles cross-encoding query-document pairs.

See: `SampleModelPackage.Reranking`

### Text Generation

Two tracks:
- **Local ONNX GenAI**: Expose `CreateGeneratorAsync()` returning an `OnnxTextGenerationTransformer` using `MLNet.TextGeneration.OnnxGenAI`.
- **Provider-agnostic MEAI**: Use any `IChatClient` provider (OpenAI, Azure, Ollama) — no model package needed.

See: `SampleModelPackage.TextGeneration`, `SampleConsumer.TextGenerationMeai`
