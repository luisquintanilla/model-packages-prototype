# CLI Reference

The Model Packages CLI tool provides commands for managing model artifacts outside of application code. It's useful for CI/CD pipelines, environment setup, and diagnostics.

## Installation

Currently run via `dotnet run`:

```bash
dotnet run --project src/ModelPackages.Tool -- <command> [options]
```

In a production release, it would ship as a .NET global tool:

```bash
dotnet tool install -g ModelPackages.Tool
model-packages <command> [options]
```

## Commands

### `prefetch`

Downloads and verifies a model, caching it locally. If the model is already cached and valid, returns the cached path immediately.

```bash
model-packages prefetch --manifest <path> [--source <name|url>] [--cache-dir <path>]
```

**Output:** Prints the local cache path to stdout on success.

**Example:**
```bash
$ model-packages prefetch --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Source resolved: 'huggingface' (from manifest default)
Download URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
Downloading from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx...
Content-Length: 86 MB
Download complete: 86 MB
Model cached at: /home/user/.local/share/ModelPackages/ModelCache/sentence-transformers/all-MiniLM-L6-v2/main/model.onnx
/home/user/.local/share/ModelPackages/ModelCache/sentence-transformers/all-MiniLM-L6-v2/main/model.onnx
```

**On subsequent runs (cached):**
```bash
$ model-packages prefetch --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Model already cached and verified at: /home/user/.local/share/ModelPackages/ModelCache/...
/home/user/.local/share/ModelPackages/ModelCache/sentence-transformers/all-MiniLM-L6-v2/main/model.onnx
```

### `verify`

Verifies the integrity of a cached model by re-computing its SHA256 hash and comparing against the manifest.

```bash
model-packages verify --manifest <path> [--source <name|url>] [--cache-dir <path>]
```

**Output:** Prints "Verification succeeded." on success.

**Example:**
```bash
$ model-packages verify --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Verification succeeded.
```

**If not cached:**
```bash
$ model-packages verify --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Verification failed: Model file not found at: /home/user/.local/share/ModelPackages/ModelCache/...
```

### `info`

Displays resolved source, cache path, and manifest metadata without downloading anything.

```bash
model-packages info --manifest <path> [--source <name|url>] [--cache-dir <path>]
```

**Example:**
```bash
$ model-packages info --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Model ID:        sentence-transformers/all-MiniLM-L6-v2
Revision:        main
File Name:       model.onnx
SHA256:          6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
Expected Bytes:  90405214
Resolved Source: huggingface (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx)
Local Path:      /home/user/.local/share/ModelPackages/ModelCache/sentence-transformers/all-MiniLM-L6-v2/main/model.onnx
```

### `clear-cache`

Removes the cached model file for the specified manifest.

```bash
model-packages clear-cache --manifest <path> [--cache-dir <path>]
```

**Example:**
```bash
$ model-packages clear-cache --manifest samples/SampleModelPackage.Onnx/model-manifest.json
Cache cleared.
```

### `cache-info`

Shows cache usage, tracked entries, and orphaned file counts. Does not require `--manifest`.

```bash
model-packages cache-info [--cache-dir <path>]
```

**Example:**
```bash
$ model-packages cache-info
Cache directory: /home/user/.cache/modelpackages
Total size:      1.2 GB
Max size:        unlimited
Entries:         11 file(s)
Orphaned:        8 file(s), 1.9 GB (run purge-cache --orphans-only to reclaim)

  microsoft/speecht5_tts/main/encoder_model.onnx               326.9 MB  Last used: 2 days ago
  ...
```

### `purge-cache`

Deletes cache files. By default deletes the entire cache directory. Use `--orphans-only` to only remove files not tracked by the cache index.

```bash
model-packages purge-cache [--orphans-only] [--yes] [--cache-dir <path>]
```

| Flag | Description |
|---|---|
| `--orphans-only` | Only delete files on disk that are not tracked by the cache index |
| `--yes`, `-y` | Skip confirmation — required to actually delete files |

**Preview what would be deleted (dry run):**
```bash
$ model-packages purge-cache --orphans-only
Found 8 orphaned file(s) (1.9 GB):
  distilbert/.../main/onnx/model.onnx                            255.5 MB
  ...
Run with --yes to delete these files.
```

**Delete orphans:**
```bash
$ model-packages purge-cache --orphans-only --yes
Purged orphaned files. Reclaimed 1.9 GB.
```

**Full cache wipe:**
```bash
$ model-packages purge-cache --yes
Cache purged. Reclaimed 3.1 GB.
```

## Global Options

| Option | Description |
|---|---|
| `--manifest <path>` | **(Required for most commands)** Path to a `model-manifest.json` file |
| `--source <name\|url>` | Override the model source. Can be a named source (e.g., `company-mirror`) or a direct URL (e.g., `https://...` or `file:///...`) |
| `--cache-dir <path>` | Override the default cache directory |
| `--orphans-only` | `purge-cache` only: delete untracked files instead of entire cache |
| `--yes`, `-y` | `purge-cache` only: skip confirmation prompt |
| `--help`, `-h` | Show usage information |

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Usage error (bad arguments, missing manifest) |
| `2` | Download failed (network error, HTTP 4xx/5xx) |
| `3` | Verification failed (SHA256 mismatch, file not found) |
| `4` | Permission error (can't write to cache dir) |
| `5` | Unexpected error |

## CI/CD Patterns

### Pre-download models in a build pipeline

```yaml
# GitHub Actions example
- name: Prefetch model
  run: |
    dotnet run --project src/ModelPackages.Tool -- \
      prefetch --manifest path/to/model-manifest.json

- name: Run tests
  run: dotnet test
```

### Verify model integrity before deployment

```yaml
- name: Verify model
  run: |
    dotnet run --project src/ModelPackages.Tool -- \
      verify --manifest path/to/model-manifest.json
```

### Use a corporate mirror in CI

```yaml
env:
  MODELPACKAGES_SOURCE: corp-mirror

- name: Prefetch from mirror
  run: |
    dotnet run --project src/ModelPackages.Tool -- \
      prefetch --manifest path/to/model-manifest.json
```

This requires a `model-sources.json` defining `corp-mirror`, or the environment variable can be set to a direct URL:

```yaml
env:
  MODELPACKAGES_SOURCE: "https://models.internal.company.com/sentence-transformers/all-MiniLM-L6-v2/onnx/model.onnx"
```

### Custom cache directory for CI

```yaml
- name: Prefetch with custom cache
  run: |
    dotnet run --project src/ModelPackages.Tool -- \
      prefetch --manifest path/to/model-manifest.json \
      --cache-dir ${{ runner.temp }}/model-cache
```

## Diagnostic Output

All diagnostic messages (progress, source resolution, download status) are written to **stderr**. Only the final result (cache path, verification status) is written to **stdout**. This means you can capture the output path in a script:

```bash
MODEL_PATH=$(model-packages prefetch --manifest manifest.json 2>/dev/null)
echo "Model is at: $MODEL_PATH"
```
