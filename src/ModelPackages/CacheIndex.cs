using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelPackages;

/// <summary>Tracks cached model files with size and access metadata for LRU eviction.</summary>
public sealed class CacheIndex
{
    private const string IndexFileName = "cache-index.json";
    private const string LockFileName = "cache-index.lock";

    [JsonIgnore]
    public long? MaxSizeBytes { get; set; }

    [JsonPropertyName("entries")]
    public List<CacheEntry> Entries { get; set; } = [];

    /// <summary>A single cached file entry.</summary>
    public sealed class CacheEntry
    {
        [JsonPropertyName("path")]
        public required string Path { get; set; }

        [JsonPropertyName("sizeBytes")]
        public long SizeBytes { get; set; }

        [JsonPropertyName("lastAccessedUtc")]
        public DateTime LastAccessedUtc { get; set; }
    }

    /// <summary>Loads the cache index from the given cache directory, or creates a new one.</summary>
    public static CacheIndex Load(string cacheDir)
    {
        var indexPath = System.IO.Path.Combine(cacheDir, IndexFileName);
        if (!File.Exists(indexPath))
            return new CacheIndex();

        try
        {
            using var _ = AcquireIndexLock(cacheDir);
            var json = File.ReadAllText(indexPath);
            return JsonSerializer.Deserialize(json, CacheIndexJsonContext.Default.CacheIndex)
                ?? new CacheIndex();
        }
        catch
        {
            return new CacheIndex();
        }
    }

    /// <summary>Saves the cache index to the given cache directory.</summary>
    public void Save(string cacheDir)
    {
        Directory.CreateDirectory(cacheDir);
        var indexPath = System.IO.Path.Combine(cacheDir, IndexFileName);
        var tempPath = indexPath + ".tmp." + Guid.NewGuid().ToString("N")[..8];

        try
        {
            using var _ = AcquireIndexLock(cacheDir);
            var json = JsonSerializer.Serialize(this, CacheIndexJsonContext.Default.CacheIndex);
            File.WriteAllText(tempPath, json);
            File.Move(tempPath, indexPath, overwrite: true);
        }
        catch
        {
            try { File.Delete(tempPath); } catch { }
            throw;
        }
    }

    /// <summary>Acquires a lock file for cache-index.json access.</summary>
    private static IDisposable AcquireIndexLock(string cacheDir)
    {
        Directory.CreateDirectory(cacheDir);
        var lockPath = System.IO.Path.Combine(cacheDir, LockFileName);
        const int maxAttempts = 50;
        const int delayMs = 100;

        for (int i = 0; i < maxAttempts; i++)
        {
            try
            {
                var fs = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
                return fs;
            }
            catch (IOException) when (i < maxAttempts - 1)
            {
                Thread.Sleep(delayMs);
            }
        }

        // Last resort: proceed without lock
        return new NoOpDisposable();
    }

    private sealed class NoOpDisposable : IDisposable { public void Dispose() { } }

    /// <summary>Records or updates a file entry in the index.</summary>
    public void Touch(string relativePath, long sizeBytes)
    {
        var entry = Entries.Find(e => string.Equals(e.Path, relativePath, StringComparison.OrdinalIgnoreCase));
        if (entry is not null)
        {
            entry.LastAccessedUtc = DateTime.UtcNow;
            entry.SizeBytes = sizeBytes;
        }
        else
        {
            Entries.Add(new CacheEntry
            {
                Path = relativePath,
                SizeBytes = sizeBytes,
                LastAccessedUtc = DateTime.UtcNow
            });
        }
    }

    /// <summary>Returns total cache size in bytes.</summary>
    [JsonIgnore]
    public long TotalSizeBytes => Entries.Sum(e => e.SizeBytes);

    /// <summary>
    /// Evicts least-recently-used entries until total size is under maxSizeBytes.
    /// Returns the list of relative paths evicted.
    /// </summary>
    public List<string> EvictIfNeeded(string cacheDir, long? maxSizeBytes, Action<string>? log = null)
    {
        var limit = maxSizeBytes ?? MaxSizeBytes;
        if (!limit.HasValue || limit.Value <= 0)
            return [];

        var evicted = new List<string>();
        var totalSize = TotalSizeBytes;

        if (totalSize <= limit.Value)
            return evicted;

        // Sort by LRU (oldest access first)
        var candidates = Entries.OrderBy(e => e.LastAccessedUtc).ToList();

        foreach (var entry in candidates)
        {
            if (totalSize <= limit.Value)
                break;

            var fullPath = Path.Combine(cacheDir, entry.Path.Replace('/', Path.DirectorySeparatorChar));
            try
            {
                if (File.Exists(fullPath))
                    File.Delete(fullPath);

                // Also clean up sidecar files
                var sidecar = fullPath + ".sha256";
                if (File.Exists(sidecar))
                    File.Delete(sidecar);
            }
            catch
            {
                // File may be locked by another process — skip
                continue;
            }

            totalSize -= entry.SizeBytes;
            evicted.Add(entry.Path);
            Entries.Remove(entry);
            log?.Invoke($"Cache eviction: removed {entry.Path} ({entry.SizeBytes / 1024 / 1024} MB, last used {entry.LastAccessedUtc:u})");
        }

        return evicted;
    }

    /// <summary>Removes entries whose files no longer exist on disk.</summary>
    public void Reconcile(string cacheDir)
    {
        Entries.RemoveAll(e =>
        {
            var fullPath = Path.Combine(cacheDir, e.Path.Replace('/', Path.DirectorySeparatorChar));
            return !File.Exists(fullPath);
        });
    }

    /// <summary>
    /// Discovers files on disk that are not tracked by the cache index.
    /// Returns a list of (relativePath, sizeBytes) for each orphaned file.
    /// Excludes sidecar (.sha256), lock (.lock), partial (.partial.*), and the index file itself.
    /// </summary>
    public List<(string RelativePath, long SizeBytes)> FindOrphanedFiles(string cacheDir)
    {
        var orphans = new List<(string, long)>();
        if (!Directory.Exists(cacheDir))
            return orphans;

        var trackedPaths = new HashSet<string>(
            Entries.Select(e => e.Path),
            StringComparer.OrdinalIgnoreCase);

        foreach (var file in Directory.EnumerateFiles(cacheDir, "*", SearchOption.AllDirectories))
        {
            var name = Path.GetFileName(file);

            // Skip metadata files
            if (name.Equals(IndexFileName, StringComparison.OrdinalIgnoreCase))
                continue;
            if (file.EndsWith(".sha256", StringComparison.OrdinalIgnoreCase))
                continue;
            if (file.EndsWith(".lock", StringComparison.OrdinalIgnoreCase))
                continue;
            if (name.Contains(".partial.", StringComparison.OrdinalIgnoreCase))
                continue;

            var relativePath = Path.GetRelativePath(cacheDir, file).Replace(Path.DirectorySeparatorChar, '/');
            if (!trackedPaths.Contains(relativePath))
                orphans.Add((relativePath, new FileInfo(file).Length));
        }

        return orphans;
    }

    /// <summary>
    /// Deletes orphaned files (on disk but not in the index), their sidecars, and empty directories.
    /// Returns the total bytes reclaimed.
    /// </summary>
    public long PurgeOrphans(string cacheDir, Action<string>? log = null)
    {
        var orphans = FindOrphanedFiles(cacheDir);
        long reclaimed = 0;

        foreach (var (relativePath, sizeBytes) in orphans)
        {
            var fullPath = Path.Combine(cacheDir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            try
            {
                if (File.Exists(fullPath))
                {
                    File.Delete(fullPath);
                    reclaimed += sizeBytes;
                    log?.Invoke($"Deleted orphan: {relativePath} ({sizeBytes / 1024 / 1024} MB)");
                }

                // Clean sidecar if present
                var sidecar = fullPath + ".sha256";
                if (File.Exists(sidecar))
                    File.Delete(sidecar);
            }
            catch
            {
                // File may be locked — skip
                continue;
            }
        }

        // Clean empty directories
        ModelCache.CleanEmptyDirectories(cacheDir);

        return reclaimed;
    }
}

[JsonSerializable(typeof(CacheIndex))]
[JsonSerializable(typeof(CacheIndex.CacheEntry))]
[JsonSerializable(typeof(List<CacheIndex.CacheEntry>))]
[JsonSourceGenerationOptions(WriteIndented = true)]
internal partial class CacheIndexJsonContext : JsonSerializerContext
{
}
