using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelPackages;

/// <summary>Tracks cached model files with size and access metadata for LRU eviction.</summary>
public sealed class CacheIndex
{
    private const string IndexFileName = "cache-index.json";

    [JsonPropertyName("maxSizeBytes")]
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
        var json = JsonSerializer.Serialize(this, CacheIndexJsonContext.Default.CacheIndex);
        File.WriteAllText(indexPath, json);
    }

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
}

[JsonSerializable(typeof(CacheIndex))]
[JsonSerializable(typeof(CacheIndex.CacheEntry))]
[JsonSerializable(typeof(List<CacheIndex.CacheEntry>))]
[JsonSourceGenerationOptions(WriteIndented = true)]
internal partial class CacheIndexJsonContext : JsonSerializerContext
{
}
