using Xunit;

namespace ModelPackages.Tests;

public class CacheTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string? _prevEnvDir;

    public CacheTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "modelpackages-test-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_tempDir);
        _prevEnvDir = Environment.GetEnvironmentVariable("MODELPACKAGES_CACHE_DIR");
        Environment.SetEnvironmentVariable("MODELPACKAGES_CACHE_DIR", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("MODELPACKAGES_CACHE_DIR", _prevEnvDir);
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }

    private static ModelManifest CreateManifest(string modelId = "test/model", string revision = "main", string filePath = "model.onnx")
    {
        var json = $$"""
        {
          "model": {
            "id": "{{modelId}}",
            "revision": "{{revision}}",
            "files": [{ "path": "{{filePath}}", "sha256": "abc", "size": 1024 }]
          },
          "sources": { "hf": { "type": "huggingface" } },
          "defaultSource": "hf"
        }
        """;
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(json));
        return ModelManifest.FromStream(stream);
    }

    [Fact]
    public void GetCachePath_CorrectStructure()
    {
        var manifest = CreateManifest();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { CacheDirOverride = _tempDir };

        var path = ModelCache.GetCachePath(manifest, file, options);

        Assert.Equal(Path.Combine(_tempDir, "test", "model", "main", "model.onnx"), path);
    }

    [Fact]
    public void GetCachePath_ModelIdWithSlashes_CreatesSubdirectories()
    {
        var manifest = CreateManifest(modelId: "org/sub/deep-model");
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { CacheDirOverride = _tempDir };

        var path = ModelCache.GetCachePath(manifest, file, options);

        Assert.Equal(
            Path.Combine(_tempDir, "org", "sub", "deep-model", "main", "model.onnx"),
            path);
    }

    [Fact]
    public void GetCachePath_CacheDirOverride_UsesOverride()
    {
        var manifest = CreateManifest();
        var file = manifest.Model.Files[0];
        var customDir = Path.Combine(_tempDir, "custom-cache");
        var options = new ModelOptions { CacheDirOverride = customDir };

        var path = ModelCache.GetCachePath(manifest, file, options);

        Assert.StartsWith(customDir, path);
    }

    [Fact]
    public void GetCachePath_EnvVarOverride_UsesEnvVar()
    {
        var envDir = Path.Combine(_tempDir, "env-cache");
        Environment.SetEnvironmentVariable("MODELPACKAGES_CACHE_DIR", envDir);

        var manifest = CreateManifest();
        var file = manifest.Model.Files[0];

        var path = ModelCache.GetCachePath(manifest, file, options: null);

        Assert.StartsWith(envDir, path);
    }

    [Fact]
    public void GetDefaultCacheDir_ReturnsNonEmpty()
    {
        var dir = ModelCache.GetDefaultCacheDir();

        Assert.NotNull(dir);
        Assert.NotEmpty(dir);
        Assert.True(Path.IsPathRooted(dir));
    }

    [Fact]
    public void GetCachePath_CustomRevision_IncludedInPath()
    {
        var manifest = CreateManifest(revision: "v2.1");
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { CacheDirOverride = _tempDir };

        var path = ModelCache.GetCachePath(manifest, file, options);

        Assert.Contains("v2.1", path);
    }

    // ── CacheIndex tests ─────────────────────────────────────────────

    [Fact]
    public void Reconcile_RemovesStaleEntries()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(cacheDir);

        // Create a file and track it, then delete the file
        var filePath = Path.Combine(cacheDir, "model.onnx");
        File.WriteAllBytes(filePath, [1, 2, 3]);

        var index = new CacheIndex();
        index.Touch("model.onnx", 3);
        index.Touch("deleted.onnx", 1000);
        index.Save(cacheDir);

        index.Reconcile(cacheDir);

        Assert.Single(index.Entries);
        Assert.Equal("model.onnx", index.Entries[0].Path);
    }

    [Fact]
    public void FindOrphanedFiles_DiscoversUntrackedFiles()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(Path.Combine(cacheDir, "org", "model", "main"));

        // Create tracked and untracked files
        var tracked = Path.Combine(cacheDir, "org", "model", "main", "tracked.onnx");
        var orphan = Path.Combine(cacheDir, "org", "model", "main", "orphan.onnx");
        File.WriteAllBytes(tracked, new byte[100]);
        File.WriteAllBytes(orphan, new byte[200]);

        var index = new CacheIndex();
        index.Touch("org/model/main/tracked.onnx", 100);

        var orphans = index.FindOrphanedFiles(cacheDir);

        Assert.Single(orphans);
        Assert.Equal("org/model/main/orphan.onnx", orphans[0].RelativePath);
        Assert.Equal(200, orphans[0].SizeBytes);
    }

    [Fact]
    public void FindOrphanedFiles_IgnoresMetadataFiles()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(cacheDir);

        // Create various metadata files that should be ignored
        File.WriteAllBytes(Path.Combine(cacheDir, "model.onnx.sha256"), [1]);
        File.WriteAllBytes(Path.Combine(cacheDir, "model.onnx.lock"), [1]);
        File.WriteAllBytes(Path.Combine(cacheDir, "model.onnx.partial.abc123"), [1]);

        var index = new CacheIndex();
        var orphans = index.FindOrphanedFiles(cacheDir);

        Assert.Empty(orphans);
    }

    [Fact]
    public void PurgeOrphans_DeletesUntrackedFilesAndSidecars()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(Path.Combine(cacheDir, "org", "model"));

        var tracked = Path.Combine(cacheDir, "org", "model", "tracked.onnx");
        var orphan = Path.Combine(cacheDir, "org", "model", "orphan.onnx");
        var orphanSidecar = orphan + ".sha256";
        File.WriteAllBytes(tracked, new byte[100]);
        File.WriteAllBytes(orphan, new byte[500]);
        File.WriteAllBytes(orphanSidecar, [1]);

        var index = new CacheIndex();
        index.Touch("org/model/tracked.onnx", 100);

        var reclaimed = index.PurgeOrphans(cacheDir);

        Assert.Equal(500, reclaimed);
        Assert.True(File.Exists(tracked));
        Assert.False(File.Exists(orphan));
        Assert.False(File.Exists(orphanSidecar));
    }

    [Fact]
    public void PurgeOrphans_CleansEmptyDirectories()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        var deepDir = Path.Combine(cacheDir, "org", "model", "main");
        Directory.CreateDirectory(deepDir);

        // Orphan in a deep directory
        File.WriteAllBytes(Path.Combine(deepDir, "orphan.onnx"), new byte[100]);

        var index = new CacheIndex();
        index.PurgeOrphans(cacheDir);

        // The empty directories should be cleaned up
        Assert.False(Directory.Exists(Path.Combine(cacheDir, "org")));
    }

    [Fact]
    public void CleanEmptyDirectories_RemovesEmptyButKeepsRoot()
    {
        var root = Path.Combine(_tempDir, "cache");
        var deep = Path.Combine(root, "a", "b", "c");
        Directory.CreateDirectory(deep);

        // One branch is empty, another has a file
        var otherDir = Path.Combine(root, "d");
        Directory.CreateDirectory(otherDir);
        File.WriteAllBytes(Path.Combine(otherDir, "keep.txt"), [1]);

        ModelCache.CleanEmptyDirectories(root);

        Assert.True(Directory.Exists(root));
        Assert.False(Directory.Exists(Path.Combine(root, "a")));
        Assert.True(Directory.Exists(otherDir));
    }

    [Fact]
    public void ClearCache_UpdatesIndex()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(Path.Combine(cacheDir, "test", "model", "main"));

        var manifest = CreateManifest();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { CacheDirOverride = cacheDir };

        // Create the cached file
        var cachePath = ModelCache.GetCachePath(manifest, file, options);
        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
        File.WriteAllBytes(cachePath, new byte[1024]);

        // Track it in the index
        var index = new CacheIndex();
        var relativePath = Path.GetRelativePath(cacheDir, cachePath).Replace(Path.DirectorySeparatorChar, '/');
        index.Touch(relativePath, 1024);
        index.Save(cacheDir);

        // ClearCache should delete the file AND update the index
        var package = ModelPackage.FromManifestStream(
            new MemoryStream(System.Text.Encoding.UTF8.GetBytes($$"""
            {
              "model": { "id": "test/model", "revision": "main", "files": [{ "path": "model.onnx", "sha256": "abc", "size": 1024 }] },
              "sources": { "hf": { "type": "huggingface" } },
              "defaultSource": "hf"
            }
            """)));
        package.ClearCache(options);

        // Reload index — entry should be gone
        var reloaded = CacheIndex.Load(cacheDir);
        Assert.Empty(reloaded.Entries);
    }

    [Fact]
    public void TotalSizeBytes_NotSerializedToJson()
    {
        var cacheDir = Path.Combine(_tempDir, "cache");
        Directory.CreateDirectory(cacheDir);

        var index = new CacheIndex();
        index.Touch("file.onnx", 1024);
        index.Save(cacheDir);

        var json = File.ReadAllText(Path.Combine(cacheDir, "cache-index.json"));
        Assert.DoesNotContain("TotalSizeBytes", json, StringComparison.OrdinalIgnoreCase);
    }
}
