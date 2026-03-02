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
}
