using Microsoft.CodeAnalysis;

namespace ModelPackages.Generators;

internal static class Diagnostics
{
    public static readonly DiagnosticDescriptor ManifestEmpty = new(
        id: "MPKG001",
        title: "Empty model manifest",
        messageFormat: "The model manifest file '{0}' is empty",
        category: "ModelPackages",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public static readonly DiagnosticDescriptor ManifestMalformed = new(
        id: "MPKG002",
        title: "Malformed model manifest",
        messageFormat: "Could not extract model ID from '{0}'. Ensure the manifest has a valid 'model.id' field.",
        category: "ModelPackages",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public static readonly DiagnosticDescriptor InvalidClassName = new(
        id: "MPKG003",
        title: "Invalid ModelPackageClassName",
        messageFormat: "The ModelPackageClassName '{0}' is not a valid C# identifier. It must be a valid identifier and not a language keyword.",
        category: "ModelPackages",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);
}
