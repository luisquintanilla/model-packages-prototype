using Xunit;

namespace ModelPackages.Tests;

[Collection("EnvVarTests")]
public class SourceResolverTests
{
    private static ModelManifest LoadFixture() =>
        ModelManifest.FromFile(Path.Combine(AppContext.BaseDirectory, "Fixtures", "valid-manifest.json"));

    [Fact]
    public void Resolve_NoOverrides_UsesManifestDefault()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];

        // Isolate from machine-level config files that could interfere
        var prevHome = Environment.GetEnvironmentVariable("USERPROFILE");
        var prevHomeUnix = Environment.GetEnvironmentVariable("HOME");
        var prevSource = Environment.GetEnvironmentVariable("MODELPACKAGES_SOURCE");
        var prevDir = Directory.GetCurrentDirectory();
        var tempDir = Path.Combine(Path.GetTempPath(), "resolver-test-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(tempDir);

        try
        {
            Environment.SetEnvironmentVariable("USERPROFILE", tempDir);
            Environment.SetEnvironmentVariable("HOME", tempDir);
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", null);
            Directory.SetCurrentDirectory(tempDir);

            var (url, sourceName) = ModelSourceResolver.Resolve(manifest, file, options: null);

            Assert.Equal("huggingface", sourceName);
            Assert.Contains("huggingface.co", url);
            Assert.Contains("sentence-transformers/all-MiniLM-L6-v2", url);
            Assert.EndsWith("/onnx/model.onnx", url);
        }
        finally
        {
            Directory.SetCurrentDirectory(prevDir);
            Environment.SetEnvironmentVariable("USERPROFILE", prevHome);
            Environment.SetEnvironmentVariable("HOME", prevHomeUnix);
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", prevSource);
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Resolve_OptionsSourceNamedSource_UsesThatSource()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { Source = "corp-mirror" };

        var (url, sourceName) = ModelSourceResolver.Resolve(manifest, file, options);

        Assert.Equal("corp-mirror", sourceName);
        Assert.StartsWith("https://models.internal.corp.com/", url);
    }

    [Fact]
    public void Resolve_OptionsSourceDirectUrl_ReturnsUrlDirectly()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { Source = "https://my-cdn.com/model.onnx" };

        var (url, sourceName) = ModelSourceResolver.Resolve(manifest, file, options);

        Assert.Equal("https://my-cdn.com/model.onnx", url);
        Assert.Equal("options-direct", sourceName);
    }

    [Fact]
    public void Resolve_OptionsSourceFileUrl_ReturnsFileUrl()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { Source = "file:///tmp/model.onnx" };

        var (url, _) = ModelSourceResolver.Resolve(manifest, file, options);

        Assert.Equal("file:///tmp/model.onnx", url);
    }

    [Fact]
    public void Resolve_EnvVarOverride_UsesEnvVar()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var prevValue = Environment.GetEnvironmentVariable("MODELPACKAGES_SOURCE");

        try
        {
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", "corp-mirror");

            var (url, sourceName) = ModelSourceResolver.Resolve(manifest, file, options: null);

            Assert.Equal("corp-mirror", sourceName);
            Assert.StartsWith("https://models.internal.corp.com/", url);
        }
        finally
        {
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", prevValue);
        }
    }

    [Fact]
    public void Resolve_OptionsSourceTakesPriorityOverEnvVar()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var prevValue = Environment.GetEnvironmentVariable("MODELPACKAGES_SOURCE");

        try
        {
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", "corp-mirror");
            var options = new ModelOptions { Source = "direct-url" };

            var (url, sourceName) = ModelSourceResolver.Resolve(manifest, file, options);

            Assert.Equal("direct-url", sourceName);
            Assert.Equal("https://example.com/model.onnx", url);
        }
        finally
        {
            Environment.SetEnvironmentVariable("MODELPACKAGES_SOURCE", prevValue);
        }
    }

    [Fact]
    public void Resolve_UnknownSourceName_ThrowsWithAvailableSources()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { Source = "nonexistent-source" };

        var ex = Assert.Throws<InvalidOperationException>(() =>
            ModelSourceResolver.Resolve(manifest, file, options));

        Assert.Contains("nonexistent-source", ex.Message);
        Assert.Contains("huggingface", ex.Message);
    }

    [Fact]
    public void Resolve_HuggingFaceSource_BuildsCorrectUrl()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];

        var (url, _) = ModelSourceResolver.Resolve(manifest, file, options: null);

        Assert.Equal(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            url);
    }

    [Fact]
    public void Resolve_MirrorSource_BuildsCorrectUrl()
    {
        var manifest = LoadFixture();
        var file = manifest.Model.Files[0];
        var options = new ModelOptions { Source = "corp-mirror" };

        var (url, _) = ModelSourceResolver.Resolve(manifest, file, options);

        Assert.Equal(
            "https://models.internal.corp.com/sentence-transformers/all-MiniLM-L6-v2/onnx/model.onnx",
            url);
    }
}
