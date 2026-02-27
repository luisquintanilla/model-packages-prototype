namespace ModelPackages;

/// <summary>
/// Resolves the download URL for a model file using the 6-level source hierarchy.
/// </summary>
internal static class ModelSourceResolver
{
    /// <summary>
    /// Resolves the download URL and source name for a model file.
    /// 
    /// Hierarchy (highest → lowest):
    /// 1. ModelOptions.Source (programmatic override)
    /// 2. Environment variable MODELPACKAGES_SOURCE
    /// 3. Project-level model-sources.json
    /// 4. User-level ~/.modelpackages/sources.json
    /// 5. Assembly metadata ModelPackagesSource
    /// 6. Manifest defaultSource
    /// </summary>
    public static (string Url, string SourceName) Resolve(
        ModelManifest manifest,
        ModelManifest.ModelFileInfo file,
        ModelOptions? options,
        Action<string>? log = null)
    {
        log ??= _ => { };

        // Load external config (levels 3 & 4)
        var (configSources, configDefault) = ModelSourceConfig.Load();

        // Determine which source name to use (6-level precedence)
        string? sourceName = null;
        string? level = null;

        // 1. ModelOptions.Source
        if (!string.IsNullOrEmpty(options?.Source))
        {
            // Could be a named source key OR a direct URL
            if (Uri.TryCreate(options.Source, UriKind.Absolute, out _) &&
                (options.Source.StartsWith("http://") || options.Source.StartsWith("https://") || options.Source.StartsWith("file://")))
            {
                // It's a direct URL — return immediately
                log($"Source resolved: direct URL from ModelOptions.Source");
                return (options.Source, "options-direct");
            }
            sourceName = options.Source;
            level = "ModelOptions.Source";
        }

        // 2. Environment variable
        if (sourceName == null)
        {
            var envSource = Environment.GetEnvironmentVariable("MODELPACKAGES_SOURCE");
            if (!string.IsNullOrEmpty(envSource))
            {
                sourceName = envSource;
                level = "env:MODELPACKAGES_SOURCE";
            }
        }

        // 3 & 4. Config files (already merged: project overrides user)
        if (sourceName == null && configDefault != null)
        {
            sourceName = configDefault;
            level = "model-sources.json";
        }

        // 5. Assembly metadata (skip for now — complex to implement generically)

        // 6. Manifest default
        if (sourceName == null)
        {
            sourceName = manifest.DefaultSource;
            level = "manifest default";
        }

        log($"Source resolved: '{sourceName}' (from {level})");

        // Now resolve the named source to a URL
        // First check config sources, then manifest sources
        ModelSource? source = null;
        if (configSources.TryGetValue(sourceName, out var configSource))
            source = configSource;

        // If not in config, check manifest sources and convert
        if (source == null && manifest.Sources.TryGetValue(sourceName, out var manifestSource))
        {
            source = ConvertManifestSource(sourceName, manifestSource);
        }

        if (source == null)
            throw new InvalidOperationException(
                $"Source '{sourceName}' not found in model-sources.json or manifest. " +
                $"Available manifest sources: [{string.Join(", ", manifest.Sources.Keys)}]. " +
                $"Set MODELPACKAGES_SOURCE or add model-sources.json.");

        var url = BuildUrl(source, manifest, file);
        log($"Download URL: {url}");
        return (url, sourceName);
    }

    private static ModelSource ConvertManifestSource(string name, ModelManifest.ManifestSource ms)
    {
        var type = ms.Type?.ToLowerInvariant() switch
        {
            "huggingface" => ModelSourceKind.HuggingFace,
            "direct" => ModelSourceKind.Direct,
            "mirror" => ModelSourceKind.Mirror,
            _ => ModelSourceKind.Direct
        };

        return new ModelSource
        {
            Name = name,
            Type = type,
            Endpoint = ms.Endpoint,
            Url = ms.Url,
            Repo = ms.Repo,
            Revision = ms.Revision
        };
    }

    private static string BuildUrl(ModelSource source, ModelManifest manifest, ModelManifest.ModelFileInfo file)
    {
        return source.Type switch
        {
            ModelSourceKind.HuggingFace => BuildHuggingFaceUrl(source, manifest, file),
            ModelSourceKind.Direct => source.Url ?? throw new InvalidOperationException(
                $"Direct source '{source.Name}' must have a Url property."),
            ModelSourceKind.Mirror => BuildMirrorUrl(source, manifest, file),
            _ => throw new InvalidOperationException($"Unknown source type: {source.Type}")
        };
    }

    private static string BuildHuggingFaceUrl(ModelSource source, ModelManifest manifest, ModelManifest.ModelFileInfo file)
    {
        var endpoint = source.Endpoint ?? "https://huggingface.co";
        var repo = source.Repo ?? manifest.Model.Id;
        var revision = source.Revision ?? manifest.Model.Revision ?? "main";
        var path = file.Path;

        return $"{endpoint.TrimEnd('/')}/{repo}/resolve/{revision}/{path}";
    }

    private static string BuildMirrorUrl(ModelSource source, ModelManifest manifest, ModelManifest.ModelFileInfo file)
    {
        var endpoint = source.Endpoint ?? throw new InvalidOperationException(
            $"Mirror source '{source.Name}' must have an Endpoint property.");
        var modelId = manifest.Model.Id;
        var path = file.Path;

        return $"{endpoint.TrimEnd('/')}/{modelId}/{path}";
    }
}
