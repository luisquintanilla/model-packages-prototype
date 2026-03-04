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
        string? primaryPath = null;

        foreach (var file in _manifest.Model.Files)
        {
            var cachePath = await EnsureFileAsync(file, options, log, cancellationToken);
            paths[file.Path] = cachePath;
            primaryPath ??= cachePath;
        }

        return new ModelFiles(paths, primaryPath!);
    }

    /// <summary>
    /// Ensures the primary model file is present locally. Downloads if missing, verifies integrity, returns absolute local path.
    /// For multi-file models, use <see cref="EnsureFilesAsync"/> if you need local paths for all files, not just the primary model file.
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
            Version: _manifest.Model.Version,
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
            IntegrityVerifier.DeleteSidecar(cachePath);
        }

        // Update cache index to remove stale entries
        try
        {
            var cacheDir = ModelCache.ResolveCacheDir(options);
            var index = CacheIndex.Load(cacheDir);
            index.Reconcile(cacheDir);
            index.Save(cacheDir);
        }
        catch { }
    }

    /// <summary>
    /// Returns cache usage information: total size, entry count, and per-entry details.
    /// </summary>
    public static CacheIndex GetCacheInfo(ModelOptions? options = null)
    {
        var cacheDir = ModelCache.ResolveCacheDir(options);
        var index = CacheIndex.Load(cacheDir);
        index.Reconcile(cacheDir);
        index.Save(cacheDir);
        return index;
    }

    /// <summary>Returns the resolved cache directory path.</summary>
    public static string GetCacheDirectory(ModelOptions? options = null)
        => ModelCache.ResolveCacheDir(options);

    /// <summary>Returns the configured maximum cache size, or null if unlimited.</summary>
    public static long? GetMaxCacheSize()
        => ModelSourceConfig.GetMaxCacheSize();

    /// <summary>
    /// Deletes cached files that are not tracked by the cache index (orphans).
    /// Returns the total bytes reclaimed.
    /// </summary>
    public static long PurgeOrphanedFiles(ModelOptions? options = null, Action<string>? log = null)
    {
        var cacheDir = ModelCache.ResolveCacheDir(options);
        var index = CacheIndex.Load(cacheDir);
        index.Reconcile(cacheDir);
        var reclaimed = index.PurgeOrphans(cacheDir, log);
        index.Save(cacheDir);
        return reclaimed;
    }

    /// <summary>
    /// Deletes the entire cache directory and all contents.
    /// Returns the total bytes deleted.
    /// </summary>
    public static long PurgeAllCache(ModelOptions? options = null, Action<string>? log = null)
    {
        var cacheDir = ModelCache.ResolveCacheDir(options);
        if (!Directory.Exists(cacheDir))
            return 0;

        long totalSize = 0;
        foreach (var file in Directory.EnumerateFiles(cacheDir, "*", SearchOption.AllDirectories))
        {
            try { totalSize += new FileInfo(file).Length; } catch { }
        }

        log?.Invoke($"Deleting entire cache directory: {cacheDir}");
        Directory.Delete(cacheDir, recursive: true);
        log?.Invoke($"Reclaimed {totalSize / 1024 / 1024} MB");
        return totalSize;
    }

    // ── Static utilities ────────────────────────────────────────────

    private static readonly string[] DefaultResourcePatterns =
        ["vocab.txt", "vocab.json", "merges.txt", "spm.model", "tokenizer.json",
         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"];

    /// <summary>
    /// Extracts embedded resources matching the given file patterns from an assembly
    /// to a model-specific cache directory, using the default cache location.
    /// Returns the directory path containing the extracted files.
    /// </summary>
    public static string ExtractResources(
        Assembly assembly,
        string modelName,
        string[]? filePatterns = null)
        => ExtractResources(assembly, modelName, options: null, filePatterns);

    /// <summary>
    /// Extracts embedded resources matching the given file patterns from an assembly
    /// to a model-specific cache directory, respecting cache directory overrides in <paramref name="options"/>.
    /// Returns the directory path containing the extracted files.
    /// </summary>
    public static string ExtractResources(
        Assembly assembly,
        string modelName,
        ModelOptions? options,
        string[]? filePatterns = null)
    {
        ArgumentNullException.ThrowIfNull(modelName);
        if (modelName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || modelName.Contains(".."))
            throw new ArgumentException($"Invalid model name '{modelName}'. Must not contain path separators or '..'.", nameof(modelName));

        filePatterns ??= DefaultResourcePatterns;

        var baseCacheDir = ModelCache.ResolveCacheDir(options);
        var cacheDir = Path.Combine(baseCacheDir, "resource-cache", modelName);
        Directory.CreateDirectory(cacheDir);

        foreach (var resourceName in assembly.GetManifestResourceNames())
        {
            var matchedFile = filePatterns
                .FirstOrDefault(p => resourceName.EndsWith(p, StringComparison.OrdinalIgnoreCase));
            if (matchedFile == null) continue;

            var targetPath = Path.Combine(cacheDir, matchedFile);
            try
            {
                using var stream = assembly.GetManifestResourceStream(resourceName)!;
                using var file = new FileStream(targetPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                stream.CopyTo(file);
            }
            catch (IOException)
            {
                // File already exists — another thread/process extracted it concurrently.
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
        var forceVerification = options?.ForceVerification ?? false;
        var progress = options?.Progress;
        var fileName = Path.GetFileName(file.Path);

        // Fast path: cached file exists and verifies
        if (!options?.ForceRedownload == true)
        {
            // Try sidecar fast-path first (unless ForceVerification is set)
            if (!forceVerification &&
                IntegrityVerifier.QuickValidate(cachePath, file.Sha256, file.Size))
            {
                log($"File already cached (sidecar verified) at: {cachePath}");
                return cachePath;
            }

            // Full SHA256 validation
            if (await IntegrityVerifier.IsValidAsync(cachePath, file.Sha256, file.Size, cancellationToken))
            {
                // Write/refresh sidecar for next time (best-effort — don't fail if write fails)
                if (!string.IsNullOrEmpty(file.Sha256))
                {
                    try { IntegrityVerifier.WriteSidecar(cachePath, file.Sha256); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
                log($"File already cached and verified at: {cachePath}");
                progress?.Report(new DownloadProgress(file.Size ?? 0, file.Size, fileName, DownloadPhase.Completed));
                UpdateCacheIndex(cachePath, file, options, log);
                return cachePath;
            }
        }

        // Resolve source URL
        progress?.Report(new DownloadProgress(0, file.Size, fileName, DownloadPhase.Resolving));
        var (url, sourceName) = ModelSourceResolver.Resolve(_manifest, file, options, log);

        // Acquire lock and download
        using (await ModelCache.AcquireLockAsync(cachePath, cancellationToken))
        {
            // Double-check after acquiring lock (another process may have downloaded)
            if (!options?.ForceRedownload == true)
            {
                if (!forceVerification &&
                    IntegrityVerifier.QuickValidate(cachePath, file.Sha256, file.Size))
                {
                    log($"File appeared in cache while waiting for lock: {cachePath}");
                    UpdateCacheIndex(cachePath, file, options, log);
                    return cachePath;
                }

                if (await IntegrityVerifier.IsValidAsync(cachePath, file.Sha256, file.Size, cancellationToken))
                {
                    if (!string.IsNullOrEmpty(file.Sha256))
                    {
                        try { IntegrityVerifier.WriteSidecar(cachePath, file.Sha256); }
                        catch (IOException) { }
                        catch (UnauthorizedAccessException) { }
                    }
                    log($"File appeared in cache while waiting for lock: {cachePath}");
                    progress?.Report(new DownloadProgress(file.Size ?? 0, file.Size, fileName, DownloadPhase.Completed));
                    return cachePath;
                }
            }

            // Atomic download: write to temp, verify, rename
            await ModelCache.AtomicWriteAsync(cachePath, async tempPath =>
            {
                await ModelDownloader.DownloadAsync(url, tempPath, options, cancellationToken);
                progress?.Report(new DownloadProgress(0, file.Size, fileName, DownloadPhase.Verifying));
                await IntegrityVerifier.VerifyAsync(tempPath, file.Sha256, file.Size, cancellationToken, log);
            }, cancellationToken);

            // Write sidecar after successful download and verification (best-effort)
            if (!string.IsNullOrEmpty(file.Sha256))
            {
                try { IntegrityVerifier.WriteSidecar(cachePath, file.Sha256); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }

        progress?.Report(new DownloadProgress(file.Size ?? 0, file.Size, fileName, DownloadPhase.Completed));
        log($"File cached at: {cachePath}");
        UpdateCacheIndex(cachePath, file, options, log);
        return cachePath;
    }

    /// <summary>Updates the cache index with access time and triggers eviction if needed.</summary>
    private void UpdateCacheIndex(string cachePath, ModelManifest.ModelFileInfo file, ModelOptions? options, Action<string> log)
    {
        try
        {
            var cacheDir = ModelCache.ResolveCacheDir(options);
            var relativePath = Path.GetRelativePath(cacheDir, cachePath).Replace(Path.DirectorySeparatorChar, '/');
            var sizeBytes = File.Exists(cachePath) ? new FileInfo(cachePath).Length : (file.Size ?? 0);

            var index = CacheIndex.Load(cacheDir);
            index.Touch(relativePath, sizeBytes);

            var maxSize = ModelSourceConfig.GetMaxCacheSize();
            index.EvictIfNeeded(cacheDir, maxSize, log);
            index.Save(cacheDir);
        }
        catch
        {
            // Cache index is best-effort; don't fail the download
        }
    }
}
