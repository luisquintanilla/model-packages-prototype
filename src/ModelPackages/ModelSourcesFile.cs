using System.Text.Json.Serialization;

namespace ModelPackages;

/// <summary>Deserialized model-sources.json configuration file.</summary>
public sealed class ModelSourcesFile
{
    /// <summary>List of configured model sources.</summary>
    [JsonPropertyName("sources")]
    public required List<ModelSourceEntry> Sources { get; init; }

    /// <summary>Key of the default source.</summary>
    [JsonPropertyName("defaultSource")]
    public string? DefaultSource { get; init; }
}

/// <summary>A single source entry in model-sources.json.</summary>
public sealed record ModelSourceEntry
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

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
