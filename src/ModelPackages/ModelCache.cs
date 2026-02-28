namespace ModelPackages;

/// <summary>
/// Manages local cache paths, atomic writes, and file-based locking for model artifacts.
/// </summary>
internal static class ModelCache
{
    private static readonly TimeSpan DefaultLockTimeout = TimeSpan.FromMinutes(15);

    /// <summary>
    /// Resolves the local cache path for a model file.
    /// Precedence: options → env MODELPACKAGES_CACHE_DIR → assembly metadata → OS default.
    /// </summary>
    public static string GetCachePath(ModelManifest manifest, ModelManifest.ModelFileInfo file, ModelOptions? options)
    {
        var baseDir = ResolveCacheDir(options);
        // Structure: {baseDir}/{modelId}/{revision}/{fileName}
        var modelId = manifest.Model.Id.Replace('/', Path.DirectorySeparatorChar);
        var revision = manifest.Model.Revision ?? "main";
        var fileName = Path.GetFileName(file.Path);
        return Path.Combine(baseDir, modelId, revision, fileName);
    }

    /// <summary>Get the OS-appropriate default cache directory.</summary>
    internal static string GetDefaultCacheDir()
    {
        if (OperatingSystem.IsWindows())
        {
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localAppData, "ModelPackages", "ModelCache");
        }
        else
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".cache", "modelpackages");
        }
    }

    private static string ResolveCacheDir(ModelOptions? options)
    {
        // 1. ModelOptions.CacheDirOverride
        if (!string.IsNullOrEmpty(options?.CacheDirOverride))
            return options.CacheDirOverride;

        // 2. Environment variable
        var envDir = Environment.GetEnvironmentVariable("MODELPACKAGES_CACHE_DIR");
        if (!string.IsNullOrEmpty(envDir))
            return envDir;

        // 3. Assembly metadata (read from calling assembly)
        // For now, skip assembly metadata resolution (complex to do reliably)

        // 4. OS default
        return GetDefaultCacheDir();
    }

    /// <summary>
    /// Atomically writes a downloaded file to the cache.
    /// Downloads to a .partial.{guid} temp file, then renames to final path.
    /// </summary>
    public static async Task AtomicWriteAsync(string finalPath, Func<string, Task> writeToTempFile, CancellationToken ct)
    {
        var dir = Path.GetDirectoryName(finalPath)!;
        Directory.CreateDirectory(dir);

        var tempPath = finalPath + $".partial.{Guid.NewGuid():N}";
        try
        {
            await writeToTempFile(tempPath);
            File.Move(tempPath, finalPath, overwrite: true);
        }
        finally
        {
            // Cleanup partial file on failure
            try { if (File.Exists(tempPath)) File.Delete(tempPath); } catch { }
        }
    }

    /// <summary>
    /// Acquires an exclusive lock file for the given cache path.
    /// Returns an IDisposable that releases the lock when disposed.
    /// Uses exponential backoff up to the timeout.
    /// </summary>
    public static async Task<IDisposable> AcquireLockAsync(string cachePath, CancellationToken ct, TimeSpan? timeout = null)
    {
        var lockPath = cachePath + ".lock";
        var dir = Path.GetDirectoryName(lockPath)!;
        Directory.CreateDirectory(dir);

        var actualTimeout = timeout ?? DefaultLockTimeout;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var delay = TimeSpan.FromMilliseconds(100);
        var maxDelay = TimeSpan.FromSeconds(5);

        while (true)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var fs = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
                return new LockHandle(fs, lockPath);
            }
            catch (IOException) when (sw.Elapsed < actualTimeout)
            {
                await Task.Delay(delay, ct);
                delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, maxDelay.TotalMilliseconds));
            }
        }
    }

    private sealed class LockHandle : IDisposable
    {
        private readonly FileStream _stream;
        private readonly string _lockPath;

        public LockHandle(FileStream stream, string lockPath)
        {
            _stream = stream;
            _lockPath = lockPath;
        }

        public void Dispose()
        {
            _stream.Dispose();
            try { File.Delete(_lockPath); } catch { }
        }
    }
}
