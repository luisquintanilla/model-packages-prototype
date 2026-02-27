using System.Text.Json;

namespace ModelPackages;

/// <summary>
/// Reads and merges model-sources.json configuration from multiple levels.
/// Resolution order (later overrides earlier): user-level → project-level.
/// </summary>
internal static class ModelSourceConfig
{
    private const string FileName = "model-sources.json";

    /// <summary>
    /// Load merged sources from project and user config files.
    /// Returns (mergedSources dict, resolvedDefaultSource string or null).
    /// </summary>
    public static (Dictionary<string, ModelSource> Sources, string? DefaultSource) Load(string? projectDir = null)
    {
        var merged = new Dictionary<string, ModelSource>(StringComparer.OrdinalIgnoreCase);
        string? defaultSource = null;

        // 1. User-level: ~/.modelpackages/model-sources.json
        var userDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".modelpackages");
        var userFile = Path.Combine(userDir, FileName);
        if (File.Exists(userFile))
        {
            var (sources, ds) = ParseFile(userFile);
            foreach (var s in sources) merged[s.Key] = s.Value;
            if (ds != null) defaultSource = ds;
        }

        // 2. Project-level: next to the .csproj (or current directory)
        var dir = projectDir ?? Directory.GetCurrentDirectory();
        var projectFile = Path.Combine(dir, FileName);
        if (File.Exists(projectFile))
        {
            var (sources, ds) = ParseFile(projectFile);
            foreach (var s in sources) merged[s.Key] = s.Value;
            if (ds != null) defaultSource = ds;
        }

        return (merged, defaultSource);
    }

    private static (Dictionary<string, ModelSource>, string?) ParseFile(string path)
    {
        var json = File.ReadAllText(path);
        var file = JsonSerializer.Deserialize(json, JsonContext.Default.ModelSourcesFile)
            ?? throw new InvalidOperationException($"Failed to deserialize {path}");

        var dict = new Dictionary<string, ModelSource>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in file.Sources)
        {
            var kind = entry.Type.ToLowerInvariant() switch
            {
                "huggingface" => ModelSourceKind.HuggingFace,
                "mirror" => ModelSourceKind.Mirror,
                "direct" => ModelSourceKind.Direct,
                _ => ModelSourceKind.Direct,
            };

            dict[entry.Name] = new ModelSource
            {
                Name = entry.Name,
                Type = kind,
                Endpoint = entry.Endpoint,
                Url = entry.Url,
                Repo = entry.Repo,
                Revision = entry.Revision,
            };
        }

        return (dict, file.DefaultSource);
    }
}
