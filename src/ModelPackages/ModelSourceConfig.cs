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
    /// Returns (mergedSources dict, resolvedDefaultSource string or null, allowedHosts set).
    /// </summary>
    public static (Dictionary<string, ModelSource> Sources, string? DefaultSource, HashSet<string> AllowedHosts) Load(string? projectDir = null)
    {
        var merged = new Dictionary<string, ModelSource>(StringComparer.OrdinalIgnoreCase);
        var allowedHosts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? defaultSource = null;

        // 1. User-level: ~/.modelpackages/model-sources.json
        var userDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".modelpackages");
        var userFile = Path.Combine(userDir, FileName);
        if (File.Exists(userFile))
        {
            var (sources, ds, clear) = ParseFile(userFile);
            if (clear)
            {
                merged.Clear();
                defaultSource = null;
            }
            foreach (var s in sources) merged[s.Key] = s.Value;
            if (ds != null) defaultSource = ds;
            MergeAllowedHosts(allowedHosts, userFile);
        }

        // 2. Project-level: next to the .csproj (or current directory)
        var dir = projectDir ?? Directory.GetCurrentDirectory();
        var projectFile = Path.Combine(dir, FileName);
        if (File.Exists(projectFile))
        {
            var (sources, ds, clear) = ParseFile(projectFile);
            if (clear)
            {
                merged.Clear();
                defaultSource = null;
            }
            foreach (var s in sources) merged[s.Key] = s.Value;
            if (ds != null) defaultSource = ds;
            MergeAllowedHosts(allowedHosts, projectFile);
        }

        return (merged, defaultSource, allowedHosts);
    }

    /// <summary>Merges allowedHosts from a config file (additive union).</summary>
    private static void MergeAllowedHosts(HashSet<string> target, string path)
    {
        var json = File.ReadAllText(path);
        var file = JsonSerializer.Deserialize(json, JsonContext.Default.ModelSourcesFile);
        if (file?.AllowedHosts is { Count: > 0 } hosts)
        {
            foreach (var host in hosts)
            {
                var trimmed = host?.Trim();
                if (!string.IsNullOrWhiteSpace(trimmed))
                    target.Add(trimmed);
            }
        }
    }

    private static (Dictionary<string, ModelSource>, string?, bool Clear) ParseFile(string path)
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

        return (dict, file.DefaultSource, file.Clear);
    }

    /// <summary>Resolves the maximum cache size from config files or environment variable.</summary>
    public static long? GetMaxCacheSize(string? projectDir = null)
    {
        // 1. Environment variable (e.g., "10GB", "500MB", or raw bytes)
        var envVal = Environment.GetEnvironmentVariable("MODELPACKAGES_CACHE_MAX_SIZE");
        if (!string.IsNullOrEmpty(envVal))
        {
            var parsed = ParseSizeString(envVal);
            if (parsed.HasValue)
                return parsed;
        }

        // 2. Config files (project overrides user)
        long? maxSize = null;

        var userDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".modelpackages");
        var userFile = Path.Combine(userDir, FileName);
        if (File.Exists(userFile))
        {
            try
            {
                var json = File.ReadAllText(userFile);
                var file = JsonSerializer.Deserialize(json, JsonContext.Default.ModelSourcesFile);
                if (file?.Cache?.MaxSizeBytes.HasValue == true)
                    maxSize = file.Cache.MaxSizeBytes;
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                // Malformed or inaccessible config — treat as no max configured
            }
        }

        var dir = projectDir ?? Directory.GetCurrentDirectory();
        var projectFile = Path.Combine(dir, FileName);
        if (File.Exists(projectFile))
        {
            try
            {
                var json = File.ReadAllText(projectFile);
                var file = JsonSerializer.Deserialize(json, JsonContext.Default.ModelSourcesFile);
                if (file?.Cache?.MaxSizeBytes.HasValue == true)
                    maxSize = file.Cache.MaxSizeBytes;
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                // Malformed or inaccessible config — treat as no max configured
            }
        }

        return maxSize;
    }

    internal static long? ParseSizeString(string value)
    {
        value = value.Trim();
        if (long.TryParse(value, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var raw))
            return raw;

        if (value.EndsWith("GB", StringComparison.OrdinalIgnoreCase) &&
            double.TryParse(value[..^2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var gb))
            return (long)(gb * 1024 * 1024 * 1024);

        if (value.EndsWith("MB", StringComparison.OrdinalIgnoreCase) &&
            double.TryParse(value[..^2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var mb))
            return (long)(mb * 1024 * 1024);

        return null;
    }
}
