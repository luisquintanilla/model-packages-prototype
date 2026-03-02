using System.Reflection;
using ModelPackages;

// Exercise key ModelPackages APIs to verify they work under NativeAOT.
Console.WriteLine("=== NativeAOT Compatibility Test ===");

// 1. Load manifest from embedded resource
var package = ModelPackage.FromManifestResource(Assembly.GetExecutingAssembly(), "AotTest.test-manifest.json");
Console.WriteLine("✓ FromManifestResource succeeded");

// 2. Get model info (no download)
var info = await package.GetModelInfoAsync(new ModelOptions
{
    Source = "test-direct",
    Logger = msg => Console.WriteLine($"  [log] {msg}")
});
Console.WriteLine($"✓ GetModelInfoAsync: {info.ModelId} rev={info.Revision}");
Console.WriteLine($"  Source: {info.ResolvedSource}");
Console.WriteLine($"  Path:   {info.LocalPath}");

// 3. Verify ModelOptions creation
var options = new ModelOptions
{
    Source = "test-direct",
    CacheDirOverride = Path.GetTempPath(),
    ForceRedownload = false,
    Logger = _ => { }
};
Console.WriteLine($"✓ ModelOptions created: CacheDirOverride={options.CacheDirOverride}");

// 4. Clear cache (no-op but exercises the code path)
package.ClearCache(options);
Console.WriteLine("✓ ClearCache succeeded");

Console.WriteLine();
Console.WriteLine("All NativeAOT compatibility checks passed.");
return 0;
