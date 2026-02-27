namespace ModelPackages;

/// <summary>Consumer-facing configuration for model operations.</summary>
public sealed record ModelOptions
{
    /// <summary>Override the model source (named source key or direct URL).</summary>
    public string? Source { get; init; }

    /// <summary>Override the cache directory (absolute path).</summary>
    public string? CacheDirOverride { get; init; }

    /// <summary>HuggingFace token for private repos. If null, reads HF_TOKEN env var.</summary>
    public string? HuggingFaceToken { get; init; }

    /// <summary>Force re-download even if cached file exists and verifies.</summary>
    public bool ForceRedownload { get; init; } = false;

    /// <summary>Logger callback for progress and status messages.</summary>
    public Action<string>? Logger { get; init; }
}
