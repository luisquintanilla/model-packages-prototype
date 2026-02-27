namespace ModelPackages;

/// <summary>Read-only information about a resolved model.</summary>
public sealed record ModelInfo(
    string ModelId,
    string Revision,
    string FileName,
    string Sha256,
    long? ExpectedBytes,
    string ResolvedSource,
    string LocalPath);
