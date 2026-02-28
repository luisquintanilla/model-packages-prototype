using System.Reflection;

namespace ModelPackages;

/// <summary>
/// Main class for fetching, caching, and verifying model artifacts.
/// Model package authors create instances via factory methods; consumers use the API methods.
/// </summary>
public sealed class ModelPackage
{
    private readonly ModelManifest _manifest;

    private ModelPackage(ModelManifest manifest)
    {
        _manifest = manifest;
    }

    // ── Factory methods ──────────────────────────────────────────────

    /// <summary>Create a ModelPackage from a manifest JSON stream.</summary>
    public static ModelPackage FromManifestStream(Stream manifestStream)
        => new(ModelManifest.FromStream(manifestStream));

    /// <summary>Create a ModelPackage from an embedded assembly resource.</summary>
    public static ModelPackage FromManifestResource(Assembly assembly, string resourceName = "model-manifest.json")
        => new(ModelManifest.FromResource(assembly, resourceName));

    /// <summary>Create a ModelPackage from a manifest file on disk.</summary>
    public static ModelPackage FromManifestFile(string path)
        => new(ModelManifest.FromFile(path));

    // ── Public API ───────────────────────────────────────────────────

    /// <summary>
    /// Ensures all model files are present locally. Downloads if missing, verifies integrity.
    /// Returns a <see cref="ModelFiles"/> with cached local paths for every file in the manifest.
    /// </summary>
    public async Task<ModelFiles> EnsureFilesAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var log = options?.Logger ?? (_ => { });
        var paths = new Dictionary<string, string>();

        foreach (var file in _manifest.Model.Files)
        {
            var cachePath = await EnsureFileAsync(file, options, log, cancellationToken);
            paths[file.Path] = cachePath;
        }

        return new ModelFiles(paths);
    }

    /// <summary>
    /// Ensures the primary model file is present locally. Downloads if missing, verifies integrity, returns absolute local path.
    /// For multi-file models, prefer <see cref="EnsureFilesAsync"/> which downloads all files.
    /// </summary>
    public async Task<string> EnsureModelAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var files = await EnsureFilesAsync(options, cancellationToken);
        return files.PrimaryModelPath;
    }

    /// <summary>
    /// Returns information about the resolved model (source, cache path, manifest metadata) without downloading.
    /// </summary>
    public Task<ModelInfo> GetModelInfoAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var file = _manifest.Model.Files[0];
        var cachePath = ModelCache.GetCachePath(_manifest, file, options);
        var (url, sourceName) = ModelSourceResolver.Resolve(_manifest, file, options);

        var info = new ModelInfo(
            ModelId: _manifest.Model.Id,
            Revision: _manifest.Model.Revision,
            FileName: Path.GetFileName(file.Path),
            Sha256: file.Sha256,
            ExpectedBytes: file.Size,
            ResolvedSource: $"{sourceName} ({url})",
            LocalPath: cachePath);

        return Task.FromResult(info);
    }

    /// <summary>
    /// Verifies all cached model files' integrity (SHA256 + optional size check).
    /// Throws if any file is missing or fails verification.
    /// </summary>
    public async Task VerifyModelAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        foreach (var file in _manifest.Model.Files)
        {
            var cachePath = ModelCache.GetCachePath(_manifest, file, options);
            await IntegrityVerifier.VerifyAsync(cachePath, file.Sha256, file.Size, cancellationToken, options?.Logger);
        }
    }

    /// <summary>
    /// Removes all cached model files.
    /// </summary>
    public void ClearCache(ModelOptions? options = null)
    {
        foreach (var file in _manifest.Model.Files)
        {
            var cachePath = ModelCache.GetCachePath(_manifest, file, options);
            if (File.Exists(cachePath))
                File.Delete(cachePath);
        }
    }

    // ── Static utilities ────────────────────────────────────────────

    private static readonly string[] DefaultResourcePatterns =
        ["vocab.txt", "vocab.json", "merges.txt", "spm.model", "tokenizer.json",
         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"];

    /// <summary>
    /// Extracts embedded resources matching the given file patterns from an assembly
    /// to a model-specific cache directory.
    /// Returns the directory path containing the extracted files.
    /// </summary>
    public static string ExtractResources(
        Assembly assembly,
        string modelName,
        string[]? filePatterns = null)
    {
        filePatterns ??= DefaultResourcePatterns;

        var cacheDir = Path.Combine(
            ModelCache.GetDefaultCacheDir(), "resource-cache", modelName);
        Directory.CreateDirectory(cacheDir);

        foreach (var resourceName in assembly.GetManifestResourceNames())
        {
            var matchedFile = filePatterns
                .FirstOrDefault(p => resourceName.EndsWith(p, StringComparison.OrdinalIgnoreCase));
            if (matchedFile == null) continue;

            var targetPath = Path.Combine(cacheDir, matchedFile);
            if (!File.Exists(targetPath))
            {
                using var stream = assembly.GetManifestResourceStream(resourceName)!;
                using var file = File.Create(targetPath);
                stream.CopyTo(file);
            }
        }

        return cacheDir;
    }

    // ── Private helpers ──────────────────────────────────────────────

    private async Task<string> EnsureFileAsync(
        ModelManifest.ModelFileInfo file,
        ModelOptions? options,
        Action<string> log,
        CancellationToken cancellationToken)
    {
        var cachePath = ModelCache.GetCachePath(_manifest, file, options);

        // Fast path: cached file exists and verifies
        if (!options?.ForceRedownload == true)
        {
            if (await IntegrityVerifier.IsValidAsync(cachePath, file.Sha256, file.Size, cancellationToken))
            {
                log($"File already cached and verified at: {cachePath}");
                return cachePath;
            }
        }

        // Resolve source URL
        var (url, sourceName) = ModelSourceResolver.Resolve(_manifest, file, options, log);

        // Acquire lock and download
        using (await ModelCache.AcquireLockAsync(cachePath, cancellationToken))
        {
            // Double-check after acquiring lock (another process may have downloaded)
            if (!options?.ForceRedownload == true)
            {
                if (await IntegrityVerifier.IsValidAsync(cachePath, file.Sha256, file.Size, cancellationToken))
                {
                    log($"File appeared in cache while waiting for lock: {cachePath}");
                    return cachePath;
                }
            }

            // Atomic download: write to temp, verify, rename
            await ModelCache.AtomicWriteAsync(cachePath, async tempPath =>
            {
                await ModelDownloader.DownloadAsync(url, tempPath, options, cancellationToken);
                await IntegrityVerifier.VerifyAsync(tempPath, file.Sha256, file.Size, cancellationToken, log);
            }, cancellationToken);
        }

        log($"File cached at: {cachePath}");
        return cachePath;
    }
}
