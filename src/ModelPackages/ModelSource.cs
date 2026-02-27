namespace ModelPackages;

/// <summary>A named model source (analogous to a NuGet package source).</summary>
public sealed class ModelSource
{
    public required string Name { get; init; }
    public required ModelSourceKind Type { get; init; }

    /// <summary>Base endpoint URL (for HuggingFace and Mirror types) or full URL (for Direct).</summary>
    public string? Endpoint { get; init; }

    /// <summary>Full URL for Direct source type.</summary>
    public string? Url { get; init; }

    /// <summary>HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").</summary>
    public string? Repo { get; init; }

    /// <summary>HuggingFace revision (branch/tag/commit). Default: "main".</summary>
    public string? Revision { get; init; }
}
