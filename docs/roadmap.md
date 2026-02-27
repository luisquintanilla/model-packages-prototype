# Roadmap

This document tracks planned improvements beyond the initial prototype. Items are grouped by priority and area.

## Phase 2: Core Improvements

### Resumable Downloads

Large model downloads (86 MB+) that fail mid-transfer currently restart from zero. Add HTTP `Range` header support to resume partial downloads from where they left off.

**Tracking issue:** [#1](https://github.com/luisquintanilla/model-packages-prototype/issues/1)

### Sidecar Hash File Caching

Every `EnsureModelAsync` call on a cached model re-computes SHA256 (~2.7s for 86 MB). Write a `.sha256` sidecar file after first verification; on subsequent loads, validate against file size + modification time + sidecar hash to reduce cached access to ~10ms.

**Tracking issue:** [#2](https://github.com/luisquintanilla/model-packages-prototype/issues/2)

### Multi-File Model Support

The manifest `files[]` array already supports listing multiple files, but the orchestrator only processes the first. Real models are often sharded (`model-00001-of-00003.onnx`) or have companion files (`tokenizer.json`, `config.json`). Extend `EnsureModelAsync` to download, cache, and verify all files, returning a directory path or a dictionary of paths.

**Tracking issue:** [#3](https://github.com/luisquintanilla/model-packages-prototype/issues/3)

### Unit Test Suite

Add a comprehensive test suite covering source resolution, cache behavior, atomic writes, integrity verification, and manifest parsing. Use a mock HTTP server for download tests.

**Tracking issue:** [#4](https://github.com/luisquintanilla/model-packages-prototype/issues/4)

### `IProgress<T>` Download Reporting

Replace the `Action<string>` logging callback with a proper `IProgress<DownloadProgress>` API that lets UIs show progress bars, percentage, bytes transferred, and estimated time remaining.

**Tracking issue:** [#5](https://github.com/luisquintanilla/model-packages-prototype/issues/5)

## Phase 2: Enterprise & Security

### Enterprise Allowed-Host Policy

Add a policy mechanism (similar to NuGet's `packageSourceMapping`) that restricts which hosts model files may be downloaded from. An enterprise config could say "only allow downloads from `models.internal.company.com`."

**Tracking issue:** [#6](https://github.com/luisquintanilla/model-packages-prototype/issues/6)

### `<clear />` Equivalent in model-sources.json

In `nuget.config`, `<clear />` removes all inherited sources. Add the same concept to `model-sources.json` so a project-level config can say "ignore all user-level and system-level sources, only use what I define here."

**Tracking issue:** [#7](https://github.com/luisquintanilla/model-packages-prototype/issues/7)

## Phase 2: Developer Experience

### Source Generator for Model Package Wrapper

The `MiniLMModel.cs` wrapper class is mostly boilerplate — it delegates to `ModelPackage` and the inference library. A C# source generator could read the embedded manifest at compile time and auto-generate the wrapper class, reducing model package authoring to just a manifest file and a `.csproj`.

**Tracking issue:** [#8](https://github.com/luisquintanilla/model-packages-prototype/issues/8)

### Cache Eviction and Size Limits

The model cache currently grows unbounded. Add configurable size limits and LRU eviction so that least-recently-used models are automatically cleaned up when the cache exceeds a threshold.

**Tracking issue:** [#9](https://github.com/luisquintanilla/model-packages-prototype/issues/9)

### NativeAOT and Trimming Compatibility

Verify that the Core SDK works correctly under `PublishTrimmed` and `PublishAot`. The SDK already uses `System.Text.Json` source generators (good for trimming), but the full pipeline needs validation.

**Tracking issue:** [#10](https://github.com/luisquintanilla/model-packages-prototype/issues/10)

## Long-Term Vision

### `dotnet model` CLI Verb

If this pattern proves valuable, the CLI tool could become a first-class .NET CLI verb — `dotnet model prefetch`, `dotnet model verify`, `dotnet model list`. This would require integration with MSBuild to discover model manifests from project references, similar to how `dotnet restore` discovers NuGet dependencies.

See [Design Decisions — Future Thinking](design-decisions.md#future-thinking-dotnet-model-cli) for more detail.
