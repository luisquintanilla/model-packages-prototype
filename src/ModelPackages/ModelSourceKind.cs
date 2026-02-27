namespace ModelPackages;

/// <summary>Type of model source.</summary>
public enum ModelSourceKind
{
    /// <summary>HuggingFace model repository. URL: {endpoint}/{repo}/resolve/{revision}/{path}</summary>
    HuggingFace,
    /// <summary>Direct URL or local file path to model artifact.</summary>
    Direct,
    /// <summary>Mirror that replicates HF directory structure. URL: {endpoint}/{modelId}/{path}</summary>
    Mirror
}
