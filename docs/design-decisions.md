# Design Decisions

This document explains the "why" behind the major design choices in this prototype. It's written for reviewers evaluating the approach, not just implementers building on top of it.

## Why not ship the model inside the NuGet package?

The most obvious question. NuGet packages *can* contain large files — there's no hard size limit. But doing so creates compounding problems:

- **Restore time scales with package size.** A 100 MB `.nupkg` means every `dotnet restore` downloads 100 MB. In CI/CD with no persistent cache, every build pays this cost. Multiply by N developers and M build agents.
- **NuGet feeds aren't model CDNs.** NuGet.org, Azure Artifacts, and GitHub Packages are optimized for code packages (typically < 5 MB). Pushing and pulling 100+ MB packages strains both their infrastructure and your bandwidth budget.
- **Version churn is expensive.** If a model is retrained weekly, each new version is another 100 MB upload + download cycle. With on-demand fetching, updating the model version means changing a hash and URL in a tiny manifest.
- **No source flexibility.** Once a model is baked into a `.nupkg`, every consumer downloads from the same NuGet feed. Enterprise environments often need to redirect to internal mirrors, air-gapped caches, or alternative model registries.

The model package approach separates **metadata** (what to download, how to verify it) from **data** (the model binary itself), just as NuGet separates package metadata from package content.

## Why separate Core SDK from inference library?

The Core SDK (`ModelPackages`) knows how to fetch, cache, and verify files. The inference library (`MLNet.Embeddings.Onnx`) knows how to build ML.NET pipelines and generate embeddings. They have no dependency on each other.

This separation matters because:

1. **Reusability across model formats.** The same Core SDK works for ONNX, safetensors, GGUF, or any future format. Only the inference layer changes.
2. **Reusability across inference frameworks.** A model package author could use TorchSharp, ONNX Runtime directly, or any other inference engine — they'd still use the same Core SDK for fetching.
3. **Testing in isolation.** Core SDK can be tested with dummy files. Inference library can be tested with local model files. Neither needs the other.
4. **Package size.** Consumers who only need fetching (e.g., a custom inference pipeline) don't pull in ML.NET dependencies.

## Why embed the manifest as an assembly resource?

The `model-manifest.json` is embedded into the model package DLL as an assembly resource (via `<EmbeddedResource>` in the `.csproj`). This survives NuGet packaging and deployment without relying on file paths.

Alternatives considered:

| Approach | Problem |
|---|---|
| Ship as a content file | Content files have unreliable paths across project types (console, web, test). The consuming app's build output directory structure isn't guaranteed. |
| Ship in a `build/` folder | Works for MSBuild integration but not for runtime access without extra targets. |
| Hardcode in C# source | Not data-driven. Can't be updated without recompiling. |
| Assembly resource ✓ | Always available via `Assembly.GetManifestResourceStream()`. Works in every deployment model (self-contained, framework-dependent, single-file, trimmed). |

**Namespace prefix gotcha:** .NET prefixes embedded resource names with the project's default namespace (e.g., `SampleModelPackage.Onnx.model-manifest.json`). The `FromManifestResource` method handles this by searching for a resource name ending with the expected filename when an exact match fails.

## Why SHA256 verification on every cache hit?

When `EnsureModelAsync` finds a cached model file, it re-computes the SHA256 hash and compares it to the manifest. This adds ~2–3 seconds for an 86 MB file. Why not skip it?

**Supply chain security.** Model poisoning is a real threat vector. A compromised model file could produce subtly wrong outputs (data exfiltration via embedding patterns, biased outputs, backdoor triggers) that are hard to detect through behavior alone. Verifying the hash on every load ensures that:

1. The cached file wasn't corrupted on disk
2. The cached file wasn't tampered with after download
3. A different process didn't replace it with a different model

For production use, this could be made configurable (e.g., "verify on first load, then trust until TTL expires"), but for a security-first default, always-verify is the right starting point.

## Why 6-level source resolution?

The source resolution hierarchy has 6 levels, which might seem excessive. Each level serves a real stakeholder:

| Level | Who | Use Case |
|---|---|---|
| `ModelOptions.Source` | App developer | "Debug with local file": `new ModelOptions { Source = "file:///C:/test/model.onnx" }` |
| `MODELPACKAGES_SOURCE` env | DevOps / CI | "All builds in this pipeline use our mirror": `env: MODELPACKAGES_SOURCE=corp-mirror` |
| Project `model-sources.json` | Team lead | "This project uses a specific source" — committed to repo |
| User `~/.modelpackages/sources.json` | Individual dev | "I always use my local cache" — personal preference |
| Assembly metadata | Package author | Bake a default into the package itself |
| Manifest `defaultSource` | Package author | The fallback if nothing else is specified |

The key principle: **higher levels never require lower levels to change.** A DevOps engineer can redirect all model downloads to a corporate mirror by setting one environment variable — without touching application code, project files, or package manifests.

This directly mirrors how `nuget.config` works with hierarchical configuration (machine → user → project → solution).

## Why `file://` URL support?

The downloader supports `file://` URIs as local file copies. This enables several workflows:

1. **Offline development.** Download the model once, point `model-sources.json` at the local copy.
2. **Air-gapped environments.** Environments without internet access can stage models on a shared drive.
3. **The PrepareModel workflow.** Track 2 requires building a `.mlnet` pipeline locally, then pointing the manifest at the output file. Without `file://`, this would require running a local HTTP server.
4. **Testing.** Integration tests can use local fixture files without mocking HTTP.

## Why atomic writes with lock files?

Two mechanisms prevent corruption:

### Atomic writes (.partial.{guid} + rename)

Downloads go to a temporary file, then are renamed to the final path. On most filesystems, rename is atomic — it either succeeds completely or not at all. This prevents:

- Partially downloaded files from being served
- Interrupted downloads from leaving corrupt cached files
- Concurrent writes from interleaving bytes

### Lock files with backoff

When multiple processes (e.g., parallel CI jobs, multiple developer machines sharing a network cache) need the same model:

1. First process creates `model.onnx.lock`
2. Others detect the lock and wait with exponential backoff (2s, 4s, 8s, ...)
3. After ~5 minutes, they break the lock (assumed stale)
4. Stale lock detection: if the lock file is >10 minutes old, it's broken automatically

This is simpler than file-based mutexes (which have cross-platform issues) and more robust than no locking (which leads to redundant downloads and potential rename races).

## Why two packaging tracks?

The prototype demonstrates two approaches to show that the architecture supports both:

### Track 1 (raw ONNX) is the flexible path
- Model author publishes: manifest + vocab (small assets only)
- Consumer's machine: downloads ONNX → builds ML.NET pipeline → runs inference
- Consumer can customize pipeline options (batch size, pooling strategy)
- Closer to how HuggingFace models are typically consumed

### Track 2 (pre-built .mlnet) is the optimized path
- Model author runs `PrepareModel` to build a self-contained `.mlnet` zip
- Consumer's machine: downloads `.mlnet` → loads pre-built pipeline → runs inference
- Faster first-run (no Fit() step), potentially smaller attack surface
- Closer to how platform-specific SDKs distribute models (CoreML, TFLite)

Both produce identical outputs. The consumer code is nearly identical. This proves the abstraction handles both cases — and opens the door for future tracks (safetensors, GGUF, etc.) without changing the Core SDK.

## Why use ML.NET's `IEstimator`/`ITransformer` pattern?

The inference library uses ML.NET's pipeline pattern (`IEstimator<ITransformer>`) rather than calling ONNX Runtime directly. This has tradeoffs:

**Benefits:**
- Composable pipeline: tokenize → ONNX → pool → normalize as separate stages
- Each stage is independently testable
- ML.NET handles batching, schema validation, and data flow
- `ModelPackager.Save/Load` serializes the entire pipeline as a self-contained zip
- Natural fit for the `IEmbeddingGenerator` abstraction from Microsoft.Extensions.AI

**Costs:**
- ML.NET is a large dependency
- The estimator/transformer pattern has a learning curve
- Pipeline construction involves `Fit()` on dummy data (for raw ONNX path)

For this prototype, the ML.NET pattern was chosen because the [reference implementation](https://github.com/luisquintanilla/mlnet-embedding-custom-transforms) already validated it. A production system might offer both ML.NET and direct ONNX Runtime paths.

## The NuGet Analogy — Mapped End to End

| NuGet Concept | Model Packages Equivalent |
|---|---|
| `.nupkg` file | Model package (small NuGet with code + manifest) |
| Package content (DLLs) | Model binary (ONNX, .mlnet, etc.) |
| `nuget.config` | `model-sources.json` |
| Package source / feed | Model source (HuggingFace, mirror, local path) |
| NuGet cache (`~/.nuget/packages`) | Model cache (`~/.local/share/ModelPackages/ModelCache`) |
| `dotnet restore` | `EnsureModelAsync()` / `prefetch` CLI |
| Package hash verification | SHA256 verification on download and cache hit |
| `dotnet nuget list source` | `info` CLI command |
| `dotnet nuget locals --clear all` | `clear-cache` CLI command |

## Future Thinking: `dotnet model` CLI

The CLI tool is currently `dotnet run --project src/ModelPackages.Tool`. In a production scenario, it would ship as a .NET global tool (`dotnet tool install -g ModelPackages.Tool`), invoked as `dotnet model-packages prefetch ...`.

Looking further ahead, if this pattern proves valuable, it could become a first-class .NET CLI verb:

```bash
dotnet model prefetch --project .    # prefetch all model dependencies
dotnet model verify --project .      # verify all cached models
dotnet model list                    # show cached models and their sources
dotnet model sources list            # show configured model sources (cf. dotnet nuget list source)
```

This would require integration with MSBuild to discover model manifests from project references — the same way `dotnet restore` discovers NuGet dependencies from `<PackageReference>` items.

The `.props`/`.targets` files in the sample model packages (e.g., `build/SampleModelPackage.Onnx.props`) are early steps toward this: they flow MSBuild properties from the model package to the consuming project.
