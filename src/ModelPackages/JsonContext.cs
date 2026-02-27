using System.Text.Json.Serialization;

namespace ModelPackages;

[JsonSerializable(typeof(ModelManifest))]
[JsonSerializable(typeof(ModelManifest.ModelIdentity))]
[JsonSerializable(typeof(ModelManifest.ModelFileInfo))]
[JsonSerializable(typeof(ModelManifest.ManifestSource))]
[JsonSerializable(typeof(Dictionary<string, ModelManifest.ManifestSource>))]
[JsonSerializable(typeof(ModelSourcesFile))]
[JsonSerializable(typeof(ModelSourceEntry))]
[JsonSerializable(typeof(List<ModelSourceEntry>))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal partial class JsonContext : JsonSerializerContext
{
}
