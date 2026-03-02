namespace ModelPackages.Generators;

/// <summary>
/// Lightweight JSON parser for extracting model ID from manifest.
/// Avoids System.Text.Json dependency (not available in netstandard2.0 source generators).
/// </summary>
internal static class ManifestParser
{
    /// <summary>Extracts the model ID from a model-manifest.json string.</summary>
    public static string? ExtractModelId(string json)
    {
        // Look for "id": "value" inside the "model" object
        // Simple approach: find "id" after "model"
        var modelIdx = json.IndexOf("\"model\"", System.StringComparison.Ordinal);
        if (modelIdx < 0)
            return null;

        var idIdx = json.IndexOf("\"id\"", modelIdx, System.StringComparison.Ordinal);
        if (idIdx < 0)
            return null;

        // Find the colon after "id"
        var colonIdx = json.IndexOf(':', idIdx + 4);
        if (colonIdx < 0)
            return null;

        // Find opening quote
        var openQuote = json.IndexOf('"', colonIdx + 1);
        if (openQuote < 0)
            return null;

        // Find closing quote
        var closeQuote = json.IndexOf('"', openQuote + 1);
        if (closeQuote < 0)
            return null;

        return json.Substring(openQuote + 1, closeQuote - openQuote - 1);
    }
}
