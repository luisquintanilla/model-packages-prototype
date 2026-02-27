# ModelPackages

Core SDK for building model packages — fetch, cache, and verify AI model artifacts from configurable sources.

Model packages are small NuGet packages (code + metadata only) that fetch large model binaries on demand, cache them locally, and verify integrity via SHA256.

## Usage

```csharp
using ModelPackages;

// Load manifest from embedded resource
var package = ModelPackage.FromManifestResource(typeof(MyModel).Assembly);

// Download, cache, and verify — returns local path
var modelPath = await package.EnsureModelAsync();
```

See the [full documentation](https://github.com/luisquintanilla/model-packages-prototype) for architecture details, authoring guides, and samples.
