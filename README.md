# Model Packages Prototype

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/luisquintanilla/model-packages-prototype)

> **What if AI models shipped like NuGet packages — small metadata packages that fetch, cache, and verify large model binaries on demand?**

## The Problem

AI models are large. A typical embedding model is 80–300 MB; LLMs run into the gigabytes. Shipping these inside NuGet packages creates real problems:

- **Bloated restores** — every `dotnet restore` re-downloads hundreds of MB
- **Storage costs** — NuGet feeds aren't designed to host large binaries at scale
- **Version control pain** — accidentally committing a model binary to git is a common mistake
- **No flexibility** — once a model is baked into a `.nupkg`, you can't redirect consumers to a corporate mirror or a local cache

But .NET developers expect things to *just work*. `dotnet add package SomeModel` should give you a working model — no manual downloads, no hunting for URLs, no SHA256 verification by hand.

## The Idea

This prototype explores a different approach: **model packages contain only code and metadata** (~few KB). The heavy model binary is fetched on first use, cached locally, and verified against a SHA256 hash. Think of it like how NuGet itself works — you don't ship source code in a package, you ship compiled artifacts that restore from feeds.

The key insight: just as `nuget.config` lets you redirect package sources (nuget.org → corporate feed → local folder), a `model-sources.json` lets you redirect model sources (HuggingFace → corporate mirror → air-gapped local path) — without changing any application code.

## Quick Start

### In GitHub Codespaces (recommended)

Click the badge above, wait for the container to build, then:

```bash
# Run the ONNX track — downloads model from HuggingFace on first run (~86 MB)
dotnet run --project samples/SampleConsumer.Onnx

# Run the .mlnet track — uses a pre-built pipeline
dotnet run --project samples/SampleConsumer.MLNet
```

### Locally

Prerequisites: [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)

```bash
git clone https://github.com/luisquintanilla/model-packages-prototype.git
cd model-packages-prototype
dotnet build
dotnet run --project samples/SampleConsumer.Onnx
```

Both samples generate text embeddings using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), compute cosine similarities, and display results — proving the full pipeline works end-to-end.

## Project Map

```
model-packages-prototype/
│
├── src/
│   ├── ModelPackages/              ← Core SDK: fetch, cache, verify (format-agnostic)
│   ├── MLNet.Embeddings.Onnx/     ← Inference library: ML.NET + MEAI wrappers
│   └── ModelPackages.Tool/        ← CLI tool: prefetch, verify, info, clear-cache
│
├── samples/
│   ├── SampleModelPackage.Onnx/   ← Model package author: raw ONNX from HuggingFace
│   ├── SampleConsumer.Onnx/       ← End user: installs package, gets embeddings
│   ├── SampleModelPackage.MLNet/  ← Model package author: pre-built .mlnet pipeline
│   └── SampleConsumer.MLNet/      ← End user: same experience, different packaging
│
├── tools/
│   └── PrepareMLNetModel/         ← Helper to build .mlnet from raw ONNX + vocab
│
└── docs/                          ← Architecture, design decisions, guides
```

## How It Works

### The 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Consumer App                                  │
│  dotnet add package SampleModelPackage.Onnx             │
│  var gen = await MiniLMModel.CreateEmbeddingGenerator() │
│  var emb = await gen.GenerateAsync(texts)               │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Model Package (authored by model publisher)   │
│  Embeds: model-manifest.json + vocab.txt                │
│  Exposes: MiniLMModel.CreateEmbeddingGeneratorAsync()   │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Inference Library (MLNet.Embeddings.Onnx)     │
│  ML.NET pipeline: tokenize → ONNX → pool → normalize   │
│  IEmbeddingGenerator<string, Embedding<float>>          │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Core SDK (ModelPackages)                      │
│  Resolve source → Check cache → Download → SHA256 verify│
│  Named sources, atomic writes, lock files               │
└─────────────────────────────────────────────────────────┘
```

1. **Core SDK** — Knows nothing about ONNX or ML.NET. It fetches files, caches them, and verifies integrity. Works with any model format.
2. **Inference Library** — Knows nothing about downloading. It builds ML.NET pipelines and wraps them as `IEmbeddingGenerator<string, Embedding<float>>` (Microsoft.Extensions.AI).
3. **Model Package** — The glue. A model author creates this as a NuGet package. It embeds a manifest (what to download, from where, expected SHA256) and small assets (tokenizer vocabulary). It references both the Core SDK and the inference library.
4. **Consumer** — Just installs the model package. One method call to get a working embedding generator. No knowledge of model sources, caching, or verification.

### Two Packaging Tracks

| | Track 1: Raw ONNX | Track 2: Pre-built .mlnet |
|---|---|---|
| **What's downloaded** | Raw `.onnx` file from HuggingFace | Pre-built `.mlnet` pipeline zip |
| **Pipeline built** | On consumer's machine (Fit) | By model author (ahead of time) |
| **First-run cost** | Download + Fit (~5s) | Download only |
| **Flexibility** | Consumer can customize pipeline | Fixed pipeline |
| **File size** | 86 MB (ONNX only) | 79 MB (ONNX + vocab + config in zip) |

The consumer code is nearly identical for both tracks — that's the proof the abstraction works.

### Named Sources (the `nuget.config` analogy)

Just as `nuget.config` redirects package feeds, `model-sources.json` redirects model sources:

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

Set `MODELPACKAGES_SOURCE=company-mirror` or drop a `model-sources.json` next to your `.csproj` — no code changes needed.

### CLI Tool

```bash
# Pre-download a model (e.g., in CI/CD)
dotnet run --project src/ModelPackages.Tool -- prefetch --manifest path/to/model-manifest.json

# Verify cached model integrity
dotnet run --project src/ModelPackages.Tool -- verify --manifest path/to/model-manifest.json

# Show resolved source and cache path
dotnet run --project src/ModelPackages.Tool -- info --manifest path/to/model-manifest.json
```

## Documentation

- **[Architecture](docs/architecture.md)** — Deep dive into the 4-layer design, manifest format, cache layout, source resolution
- **[Design Decisions](docs/design-decisions.md)** — The "why" behind every major choice
- **[Authoring Guide](docs/authoring-guide.md)** — How to create your own model package
- **[CLI Reference](docs/cli-reference.md)** — All commands, options, exit codes, CI/CD patterns

## Technology

- **.NET 10** (preview)
- **ML.NET** — Pipeline construction, ONNX Runtime integration
- **Microsoft.Extensions.AI** — `IEmbeddingGenerator<string, Embedding<float>>` abstraction
- **OnnxRuntime** — Model inference
- **System.Text.Json** — Source-generated serialization for manifests

## Status

This is a **prototype / proof of concept** exploring the design space. Not production-ready.

## License

TBD
