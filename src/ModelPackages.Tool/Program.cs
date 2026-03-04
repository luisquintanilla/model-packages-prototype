using ModelPackages;

return await RunAsync(args);

static async Task<int> RunAsync(string[] args)
{
    if (args.Length == 0)
    {
        PrintUsage();
        return 1;
    }

    var command = args[0];

    if (command is "--help" or "-h")
    {
        PrintUsage();
        return 0;
    }

    string? manifest = null;
    string? source = null;
    string? cacheDir = null;

    for (int i = 1; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--manifest" when i + 1 < args.Length:
                manifest = args[++i];
                break;
            case "--source" when i + 1 < args.Length:
                source = args[++i];
                break;
            case "--cache-dir" when i + 1 < args.Length:
                cacheDir = args[++i];
                break;
            default:
                Console.Error.WriteLine($"Unknown argument: {args[i]}");
                PrintUsage();
                return 1;
        }
    }

    var options = new ModelOptions
    {
        Source = source,
        CacheDirOverride = cacheDir,
        Logger = msg => Console.Error.WriteLine(msg),
    };

    // Commands that don't require a manifest
    if (command == "cache-info")
    {
        CacheInfo(options);
        return 0;
    }

    if (manifest is null)
    {
        Console.Error.WriteLine("Error: --manifest <path> is required.");
        PrintUsage();
        return 1;
    }

    if (!File.Exists(manifest))
    {
        Console.Error.WriteLine($"Error: Manifest file not found: {manifest}");
        return 1;
    }

    try
    {
        var package = ModelPackage.FromManifestFile(manifest);

        switch (command)
        {
            case "prefetch":
                return await PrefetchAsync(package, options);
            case "verify":
                return await VerifyAsync(package, options);
            case "info":
                return await InfoAsync(package, options);
            case "clear-cache":
                ClearCache(package, options);
                return 0;
            default:
                Console.Error.WriteLine($"Unknown command: {command}");
                PrintUsage();
                return 1;
        }
    }
    catch (UnauthorizedAccessException ex)
    {
        Console.Error.WriteLine($"Permission error: {ex.Message}");
        return 4;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Unexpected error: {ex.Message}");
        return 5;
    }
}

static async Task<int> PrefetchAsync(ModelPackage package, ModelOptions options)
{
    try
    {
        var path = await package.EnsureModelAsync(options);
        Console.WriteLine(path);
        return 0;
    }
    catch (HttpRequestException ex)
    {
        Console.Error.WriteLine($"Download failed: {ex.Message}");
        return 2;
    }
    catch (InvalidOperationException ex)
    {
        Console.Error.WriteLine($"Verification failed: {ex.Message}");
        return 3;
    }
}

static async Task<int> VerifyAsync(ModelPackage package, ModelOptions options)
{
    try
    {
        await package.VerifyModelAsync(options);
        Console.WriteLine("Verification succeeded.");
        return 0;
    }
    catch (FileNotFoundException ex)
    {
        Console.Error.WriteLine($"Verification failed: {ex.Message}");
        return 3;
    }
    catch (InvalidOperationException ex)
    {
        Console.Error.WriteLine($"Verification failed: {ex.Message}");
        return 3;
    }
}

static async Task<int> InfoAsync(ModelPackage package, ModelOptions options)
{
    var info = await package.GetModelInfoAsync(options);
    Console.WriteLine($"Model ID:        {info.ModelId}");
    if (info.Version != null)
        Console.WriteLine($"Version:         {info.Version}");
    Console.WriteLine($"Revision:        {info.Revision}");
    Console.WriteLine($"File Name:       {info.FileName}");
    Console.WriteLine($"SHA256:          {info.Sha256}");
    Console.WriteLine($"Expected Bytes:  {(info.ExpectedBytes.HasValue ? info.ExpectedBytes.Value.ToString() : "unknown")}");
    Console.WriteLine($"Resolved Source: {info.ResolvedSource}");
    Console.WriteLine($"Local Path:      {info.LocalPath}");
    return 0;
}

static void ClearCache(ModelPackage package, ModelOptions options)
{
    package.ClearCache(options);
    Console.WriteLine("Cache cleared.");
}

static void CacheInfo(ModelOptions options)
{
    var index = ModelPackage.GetCacheInfo(options);
    var cacheDir = ModelPackage.GetCacheDirectory(options);
    var maxSize = ModelPackage.GetMaxCacheSize();

    Console.WriteLine($"Cache directory: {cacheDir}");
    Console.WriteLine($"Total size:      {FormatSize(index.TotalSizeBytes)}");
    Console.WriteLine($"Max size:        {(maxSize.HasValue ? FormatSize(maxSize.Value) + " (configured)" : "unlimited")}");
    Console.WriteLine($"Entries:         {index.Entries.Count} file(s)");

    if (index.Entries.Count > 0)
    {
        Console.WriteLine();
        foreach (var entry in index.Entries.OrderByDescending(e => e.LastAccessedUtc))
        {
            var ago = DateTime.UtcNow - entry.LastAccessedUtc;
            var agoStr = ago.TotalHours < 1 ? $"{ago.TotalMinutes:F0} min ago"
                : ago.TotalDays < 1 ? $"{ago.TotalHours:F0} hours ago"
                : $"{ago.TotalDays:F0} days ago";
            Console.WriteLine($"  {entry.Path,-60} {FormatSize(entry.SizeBytes),10}  Last used: {agoStr}");
        }
    }
}

static string FormatSize(long bytes)
{
    if (bytes >= 1024L * 1024 * 1024)
        return $"{bytes / 1024.0 / 1024 / 1024:F1} GB";
    if (bytes >= 1024 * 1024)
        return $"{bytes / 1024.0 / 1024:F1} MB";
    if (bytes >= 1024)
        return $"{bytes / 1024.0:F1} KB";
    return $"{bytes} B";
}

static void PrintUsage()
{
    Console.Error.WriteLine("""
        Usage: model-packages <command> [--manifest <path>] [options]

        Commands:
          prefetch      Download and verify the model
          verify        Verify cached model integrity
          info          Show resolved source, cache path, manifest metadata
          clear-cache   Remove cached model
          cache-info    Show cache usage and entries (no --manifest required)

        Options:
          --manifest <path>    Path to model-manifest.json (required for most commands)
          --source <name|url>  Override model source
          --cache-dir <path>   Override cache directory

        Exit codes:
          0  Success
          1  Usage error
          2  Download failed
          3  Verification failed
          4  Permission error
          5  Unexpected error
        """);
}
