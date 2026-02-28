namespace ModelPackages;

/// <summary>
/// Result of <see cref="ModelPackage.EnsureFilesAsync"/> — contains cached local paths for all manifest files.
/// </summary>
public sealed class ModelFiles
{
    private readonly IReadOnlyDictionary<string, string> _files;

    internal ModelFiles(IReadOnlyDictionary<string, string> files)
    {
        _files = files;
    }

    /// <summary>All cached file paths, keyed by their manifest path (e.g., "onnx/model.onnx").</summary>
    public IReadOnlyDictionary<string, string> Files => _files;

    /// <summary>Path to the primary model file (first entry in the manifest).</summary>
    public string PrimaryModelPath => _files.Values.First();

    /// <summary>Directory containing the primary model file.</summary>
    public string ModelDirectory => Path.GetDirectoryName(PrimaryModelPath)!;

    /// <summary>Gets the cached local path for a specific manifest file path.</summary>
    public string GetPath(string manifestPath)
    {
        if (_files.TryGetValue(manifestPath, out var localPath))
            return localPath;
        throw new KeyNotFoundException(
            $"File '{manifestPath}' not found in manifest. Available: [{string.Join(", ", _files.Keys)}]");
    }

    /// <summary>Checks whether a file path exists in the manifest.</summary>
    public bool HasFile(string manifestPath) => _files.ContainsKey(manifestPath);
}
