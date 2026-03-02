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
}
