# Architecture

This document describes the internal architecture of the Model Packages prototype. It's written for someone who wants to understand how the system works, not just how to use it.

## Overview

The system has a single job: let .NET developers consume AI models from NuGet packages without shipping large binary files inside those packages. The model binary is fetched on demand, cached locally, verified against a SHA256 hash, and then used for inference.

The architecture has **four layers**, each with a clear responsibility and no knowledge of the layers above it.

## The Four Layers

```
┌──────────────────────────────────────────┐
│  Layer 4: Consumer App                   │  "I just want embeddings"
├──────────────────────────────────────────┤
│  Layer 3: Model Package                  │  "I know which model and how to wire it"
├──────────────────────────────────────────┤
│  Layer 2: Inference Library              │  "I know how to run ONNX models"
├──────────────────────────────────────────┤
│  Layer 1: Core SDK                       │  "I fetch, cache, and verify files"
└──────────────────────────────────────────┘
```

### Layer 1: Core SDK (`ModelPackages`)

**What it does:** Resolves where to download a model from, downloads it with streaming and retries, caches it on the local filesystem, and verifies integrity via SHA256.

**What it doesn't know:** What the file *is*. It could be an ONNX model, a .mlnet pipeline, a safetensors file, or a GGUF — the Core SDK doesn't care. It's a generic fetch-cache-verify engine.

**Key classes:**

| Class | Responsibility |
|---|---|
| `ModelPackage` | Orchestrator — the public API. Factory methods (`FromManifestResource`, `FromManifestFile`) + operations (`EnsureModelAsync`, `VerifyModelAsync`, etc.) |
| `ModelManifest` | Parses `model-manifest.json`. Contains model identity, file info (path, SHA256, size), named sources, and default source. |
| `ModelSourceResolver` | Resolves which source to use via 6-level hierarchy. Returns a download URL. |
| `ModelSourceConfig` | Reads and merges `model-sources.json` files from project and user directories. |
| `ModelCache` | Computes cache paths, manages atomic writes (temp file + rename), and handles lock files with backoff. |
| `ModelDownloader` | HTTP streaming download with retries, HuggingFace auth, `file://` local copy. |
| `IntegrityVerifier` | Streaming SHA256 computation and size verification. |

**The download flow:**

```
EnsureModelAsync(options)
    │
    ├─ Is model already cached and valid? ──yes──→ return cached path
    │
    ├─ Acquire lock file (with backoff)
    │
    ├─ Double-check cache (another process may have downloaded)
    │
    ├─ Resolve source URL (6-level hierarchy)
    │
    ├─ Download to temp file (.partial.{guid})
    │
    ├─ Verify SHA256 + size on temp file
    │
    ├─ Atomic rename: temp → final path
    │
    ├─ Release lock
    │
    └─ return final path
```

### Layer 2: Inference Library (`MLNet.Embeddings.Onnx`)

**What it does:** Builds an ML.NET pipeline that tokenizes text, runs it through an ONNX model, applies pooling and normalization, and returns embeddings. Also wraps the pipeline as `IEmbeddingGenerator<string, Embedding<float>>` from Microsoft.Extensions.AI.

**What it doesn't know:** Where the model file came from. It receives a local file path and builds a pipeline from it.

This layer is ported from the [reference implementation](https://github.com/luisquintanilla/mlnet-embedding-custom-transforms) and includes:

- `OnnxTextEmbeddingEstimator` / `OnnxTextEmbeddingTransformer` — ML.NET custom estimator/transformer
- `WordPieceTokenizer` / `WordPieceTokenizerTransformer` — Tokenization with special tokens
- `PoolingTransformer` — Mean/CLS pooling strategies
- `NormalizationTransformer` — L2 normalization
- `OnnxEmbeddingGenerator` — MEAI `IEmbeddingGenerator` wrapper
- `ModelPackager` — Save/load self-contained `.mlnet` zip files

### Layer 3: Model Package (authored by model publisher)

**What it does:** Ties a specific model to the Core SDK and inference library. Embeds a `model-manifest.json` (as an assembly resource) and any small assets (e.g., `vocab.txt` for tokenization). Exposes a simple public API.

**What it creates:** A NuGet package (~200 KB) that contains:
- `model-manifest.json` — what to download, from where, expected hash
- `vocab.txt` — tokenizer vocabulary (small enough to embed)
- `MiniLMModel.cs` — public API: `CreateEmbeddingGeneratorAsync()`
- Project references to Core SDK + inference library

**Example — the entire public API surface for a model package:**

```csharp
public static class MiniLMModel
{
    public static Task<IEmbeddingGenerator<string, Embedding<float>>>
        CreateEmbeddingGeneratorAsync(ModelOptions? options = null, CancellationToken ct = default);

    public static Task<string> EnsureModelAsync(ModelOptions? options = null, CancellationToken ct = default);
    public static Task<ModelInfo> GetModelInfoAsync(ModelOptions? options = null, CancellationToken ct = default);
    public static Task VerifyModelAsync(ModelOptions? options = null, CancellationToken ct = default);
}
```

### Layer 4: Consumer (end-user application)

**What it does:** Installs the model package and uses it. No knowledge of model sources, caching, verification, or ONNX.

```csharp
using SampleModelPackage.Onnx;

var generator = await MiniLMModel.CreateEmbeddingGeneratorAsync();
var embeddings = await generator.GenerateAsync(new[] { "Hello world" });
```

That's it. The model is downloaded, cached, verified, and loaded transparently.

## Manifest Format

The manifest is the contract between the model author and the Core SDK. It describes what to download, how to verify it, and where to find it.

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
      }
    ]
  },
  "sources": {
    "huggingface": {
      "type": "huggingface"
    },
    "company-mirror": {
      "type": "mirror",
      "endpoint": "https://models.internal.company.com"
    },
    "local-dev": {
      "type": "direct",
      "url": "file:///C:/models/model.onnx"
    }
  },
  "defaultSource": "huggingface"
}
```

**Fields:**

| Field | Purpose |
|---|---|
| `model.id` | Model identifier (typically `org/model-name` for HuggingFace) |
| `model.revision` | Git revision or version tag |
| `model.files[].path` | Path within the source (used to build download URL for HuggingFace/mirror sources) |
| `model.files[].sha256` | Expected SHA256 hash of the downloaded file |
| `model.files[].size` | Expected file size in bytes |
| `sources` | Named sources — each has a type and type-specific configuration |
| `defaultSource` | Which source to use when no override is specified |

**Source types:**

| Type | Behavior | URL Pattern |
|---|---|---|
| `huggingface` | Standard HuggingFace CDN | `{endpoint}/{repo}/resolve/{revision}/{path}` |
| `direct` | Exact URL (HTTP, HTTPS, or `file://`) | Uses `url` field directly |
| `mirror` | Corporate mirror with standard path layout | `{endpoint}/{model.id}/{path}` |

## Source Resolution Hierarchy

When the Core SDK needs to download a model, it determines *which source to use* via a 6-level hierarchy. This is the core of the `nuget.config` analogy — it lets different environments redirect model downloads without changing any code.

```
Priority 1 (highest): ModelOptions.Source       ← programmatic override
Priority 2:           MODELPACKAGES_SOURCE env  ← environment variable
Priority 3:           project model-sources.json ← next to .csproj
Priority 4:           user ~/.modelpackages/sources.json ← user-global
Priority 5:           assembly metadata         ← baked into the package
Priority 6 (lowest):  manifest defaultSource    ← author's default
```

**Why 6 levels?** Different stakeholders have different needs:
- **App developer** (level 1–2): "I want to test with a local model file"
- **DevOps** (level 2): "All builds must use our corporate mirror"
- **Project config** (level 3): "This project uses a specific source"
- **User preference** (level 4): "I always want to use my local cache"
- **Package author** (level 5–6): "Default to HuggingFace unless overridden"

## Cache Layout

Models are cached under the user's local application data directory:

```
%LOCALAPPDATA%/ModelPackages/ModelCache/          (Windows)
~/.local/share/ModelPackages/ModelCache/           (Linux/macOS)

ModelCache/
└── sentence-transformers/
    └── all-MiniLM-L6-v2/
        └── main/
            ├── model.onnx              ← Track 1 cached artifact
            └── pipeline.mlnet          ← Track 2 cached artifact
```

**Path structure:** `{model.id}/{revision}/{filename}`

This mirrors the HuggingFace cache layout and ensures different models and revisions never collide.

### Atomic Writes

To prevent partial/corrupt files from being served (e.g., if a download is interrupted or two processes download simultaneously):

1. Download goes to a temp file: `model.onnx.partial.{guid}`
2. SHA256 is verified on the temp file
3. If valid: atomic `File.Move(temp, final, overwrite: true)`
4. If invalid: temp file is deleted, exception thrown

### Lock Files

When multiple processes need the same model simultaneously:

1. First process creates `model.onnx.lock`
2. Other processes detect the lock and poll with exponential backoff (up to 5 minutes)
3. Lock is released after download completes (or on failure)
4. Stale locks (>10 minutes old) are automatically broken

## Two Packaging Tracks

This prototype demonstrates two approaches to packaging the same model:

### Track 1: Raw ONNX from HuggingFace

```
Author creates:                    Consumer's machine does:
model-manifest.json ──────────→ Download .onnx from HuggingFace
vocab.txt (embedded) ─────────→ Build ML.NET pipeline (Fit)
                                 Wrap as IEmbeddingGenerator
```

- The consumer's machine runs `Fit()` to build the ML.NET pipeline
- Flexible: consumer can customize pipeline options
- Requires OnnxRuntime native binaries in the consumer app

### Track 2: Pre-built .mlnet Pipeline

```
Author runs PrepareModel:         Consumer's machine does:
raw .onnx + vocab ───→ .mlnet    Download .mlnet from source
                                  Load pre-built pipeline
                                  Wrap as IEmbeddingGenerator
```

- Author runs `PrepareModel.BuildAndSaveAsync()` to build a self-contained `.mlnet` zip
- The `.mlnet` contains: ONNX model + vocab + config + manifest (all in one zip)
- Consumer just loads it — no `Fit()` step
- Less flexible, but faster first-run and simpler consumer setup

### Proof of Architecture

Both tracks produce **byte-identical embeddings** for the same inputs. The consumer code is nearly identical — only the `using` statement differs. This demonstrates that the 4-layer architecture successfully abstracts away the packaging strategy.
