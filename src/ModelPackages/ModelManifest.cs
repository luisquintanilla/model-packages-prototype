using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelPackages;

/// <summary>Deserialized model-manifest.json.</summary>
public sealed class ModelManifest
{
    /// <summary>Model identity block.</summary>
    [JsonPropertyName("model")]
    public required ModelIdentity Model { get; init; }

    /// <summary>Named sources keyed by source name.</summary>
    [JsonPropertyName("sources")]
    public required Dictionary<string, ManifestSource> Sources { get; init; }

    /// <summary>Key of the default source in <see cref="Sources"/>.</summary>
    [JsonPropertyName("defaultSource")]
    public required string DefaultSource { get; init; }

    // ── Factory methods ──────────────────────────────────────────────

    /// <summary>Deserialize a manifest from a JSON stream.</summary>
    public static ModelManifest FromStream(Stream stream)
    {
        return JsonSerializer.Deserialize(stream, JsonContext.Default.ModelManifest)
            ?? throw new InvalidOperationException("Failed to deserialize model manifest.");
    }

    /// <summary>Load a manifest from an embedded resource.</summary>
    public static ModelManifest FromResource(Assembly assembly, string resourceName = "model-manifest.json")
    {
        // Try exact name first, then search by suffix (handles namespace-prefixed names)
        var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream == null)
        {
            var match = assembly.GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith("." + resourceName, StringComparison.OrdinalIgnoreCase)
                                  || n.Equals(resourceName, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                stream = assembly.GetManifestResourceStream(match);
        }
        if (stream == null)
            throw new FileNotFoundException(
                $"Embedded resource '{resourceName}' not found in assembly '{assembly.FullName}'. " +
                $"Available: [{string.Join(", ", assembly.GetManifestResourceNames())}]");
        using (stream)
            return FromStream(stream);
    }

    /// <summary>Load a manifest from a file on disk.</summary>
    public static ModelManifest FromFile(string path)
    {
        using var stream = File.OpenRead(path);
        return FromStream(stream);
    }

    // ── Nested types ─────────────────────────────────────────────────

    /// <summary>Identity of the model (id, revision, file list).</summary>
    public sealed record ModelIdentity
    {
        [JsonPropertyName("id")]
        public required string Id { get; init; }

        [JsonPropertyName("revision")]
        public required string Revision { get; init; }

        [JsonPropertyName("files")]
        public required IReadOnlyList<ModelFileInfo> Files { get; init; }
    }

    /// <summary>A single file entry inside the manifest.</summary>
    public sealed record ModelFileInfo
    {
        [JsonPropertyName("path")]
        public required string Path { get; init; }

        [JsonPropertyName("sha256")]
        public required string Sha256 { get; init; }

        [JsonPropertyName("size")]
        public long? Size { get; init; }
    }

    /// <summary>A source entry inside the manifest sources block.</summary>
    public sealed record ManifestSource
    {
        [JsonPropertyName("type")]
        public required string Type { get; init; }

        [JsonPropertyName("endpoint")]
        public string? Endpoint { get; init; }

        [JsonPropertyName("url")]
        public string? Url { get; init; }

        [JsonPropertyName("repo")]
        public string? Repo { get; init; }

        [JsonPropertyName("revision")]
        public string? Revision { get; init; }
    }
}
