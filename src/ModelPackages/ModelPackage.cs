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
    /// Ensures the model is present locally. Downloads if missing, verifies integrity, returns absolute local path.
    /// </summary>
    public async Task<string> EnsureModelAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var file = _manifest.Model.Files[0]; // MVP: single file
        var log = options?.Logger ?? (_ => { });

        // Resolve cache path
        var cachePath = ModelCache.GetCachePath(_manifest, file, options);

        // Fast path: cached file exists and verifies
        if (!options?.ForceRedownload == true)
        {
            if (await IntegrityVerifier.IsValidAsync(cachePath, file.Sha256, file.Size, cancellationToken))
            {
                log($"Model already cached and verified at: {cachePath}");
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
                    log($"Model appeared in cache while waiting for lock: {cachePath}");
                    return cachePath;
                }
            }

            // Atomic download: write to temp, verify, rename
            await ModelCache.AtomicWriteAsync(cachePath, async tempPath =>
            {
                await ModelDownloader.DownloadAsync(url, tempPath, options, cancellationToken);
                await IntegrityVerifier.VerifyAsync(tempPath, file.Sha256, file.Size, cancellationToken);
            }, cancellationToken);
        }

        log($"Model cached at: {cachePath}");
        return cachePath;
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
    /// Verifies the cached model's integrity (SHA256 + optional size check).
    /// Throws if the file is missing or fails verification.
    /// </summary>
    public async Task VerifyModelAsync(
        ModelOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var file = _manifest.Model.Files[0];
        var cachePath = ModelCache.GetCachePath(_manifest, file, options);
        await IntegrityVerifier.VerifyAsync(cachePath, file.Sha256, file.Size, cancellationToken);
    }

    /// <summary>
    /// Removes the cached model file.
    /// </summary>
    public void ClearCache(ModelOptions? options = null)
    {
        var file = _manifest.Model.Files[0];
        var cachePath = ModelCache.GetCachePath(_manifest, file, options);
        if (File.Exists(cachePath))
            File.Delete(cachePath);
    }
}
