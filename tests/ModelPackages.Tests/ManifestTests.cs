using System.Text;
using System.Text.Json;
using Xunit;

namespace ModelPackages.Tests;

public class ManifestTests
{
    private static string FixturePath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    [Fact]
    public void FromFile_ValidManifest_ParsesCorrectly()
    {
        var manifest = ModelManifest.FromFile(FixturePath("valid-manifest.json"));

        Assert.Equal("sentence-transformers/all-MiniLM-L6-v2", manifest.Model.Id);
        Assert.Equal("main", manifest.Model.Revision);
        Assert.Single(manifest.Model.Files);
        Assert.Equal("onnx/model.onnx", manifest.Model.Files[0].Path);
        Assert.Equal("6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452", manifest.Model.Files[0].Sha256);
        Assert.Equal(90405214L, manifest.Model.Files[0].Size);
        Assert.Equal("huggingface", manifest.DefaultSource);
    }

    [Fact]
    public void FromFile_MultiFileManifest_AllFilesPresent()
    {
        var manifest = ModelManifest.FromFile(FixturePath("multi-file-manifest.json"));

        Assert.Equal(3, manifest.Model.Files.Count);
        Assert.Equal("model-00001-of-00003.onnx", manifest.Model.Files[0].Path);
        Assert.Equal("model-00002-of-00003.onnx", manifest.Model.Files[1].Path);
        Assert.Equal("tokenizer.json", manifest.Model.Files[2].Path);
        Assert.Equal("v1.0", manifest.Model.Revision);
    }

    [Fact]
    public void FromFile_SizeZero_ParsedAsZero()
    {
        var manifest = ModelManifest.FromFile(FixturePath("no-size-manifest.json"));

        Assert.Equal(0L, manifest.Model.Files[0].Size);
    }

    [Fact]
    public void FromStream_ValidJson_Succeeds()
    {
        var json = """
        {
          "model": {
            "id": "test/model",
            "revision": "main",
            "files": [{ "path": "model.bin", "sha256": "abc123", "size": 1024 }]
          },
          "sources": { "hf": { "type": "huggingface" } },
          "defaultSource": "hf"
        }
        """;
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(json));

        var manifest = ModelManifest.FromStream(stream);

        Assert.Equal("test/model", manifest.Model.Id);
        Assert.Equal(1024L, manifest.Model.Files[0].Size);
    }

    [Fact]
    public void FromStream_InvalidJson_ThrowsJsonException()
    {
        using var stream = new MemoryStream("not json"u8.ToArray());

        Assert.Throws<JsonException>(() => ModelManifest.FromStream(stream));
    }

    [Fact]
    public void FromFile_NonExistent_ThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ModelManifest.FromFile("does-not-exist.json"));
    }

    [Fact]
    public void FromFile_MultipleSources_AllParsed()
    {
        var manifest = ModelManifest.FromFile(FixturePath("valid-manifest.json"));

        Assert.Equal(3, manifest.Sources.Count);
        Assert.True(manifest.Sources.ContainsKey("huggingface"));
        Assert.True(manifest.Sources.ContainsKey("corp-mirror"));
        Assert.True(manifest.Sources.ContainsKey("direct-url"));
        Assert.Equal("huggingface", manifest.Sources["huggingface"].Type);
        Assert.Equal("mirror", manifest.Sources["corp-mirror"].Type);
        Assert.Equal("https://models.internal.corp.com", manifest.Sources["corp-mirror"].Endpoint);
    }

    [Fact]
    public void FromFile_SizeOmitted_NullSize()
    {
        var json = """
        {
          "model": {
            "id": "test/model",
            "revision": "main",
            "files": [{ "path": "model.bin", "sha256": "abc" }]
          },
          "sources": { "hf": { "type": "huggingface" } },
          "defaultSource": "hf"
        }
        """;
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, json);
            var manifest = ModelManifest.FromFile(tempFile);
            Assert.Null(manifest.Model.Files[0].Size);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }
}
