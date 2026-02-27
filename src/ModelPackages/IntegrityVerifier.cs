using System.Security.Cryptography;

namespace ModelPackages;

/// <summary>
/// Verifies model file integrity via streaming SHA256 and optional size checks.
/// </summary>
internal static class IntegrityVerifier
{
    private const int BufferSize = 81920; // 80KB chunks

    /// <summary>
    /// Verifies a file's SHA256 hash matches the expected value.
    /// Optionally checks file size.
    /// On mismatch, deletes the file and throws with actionable message.
    /// </summary>
    public static async Task VerifyAsync(
        string filePath,
        string expectedSha256,
        long? expectedSize,
        CancellationToken ct)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found at: {filePath}");

        // Optional size check (fast, before hashing)
        if (expectedSize.HasValue)
        {
            var actualSize = new FileInfo(filePath).Length;
            if (actualSize != expectedSize.Value)
            {
                File.Delete(filePath);
                throw new InvalidOperationException(
                    $"Size mismatch for {filePath}. Expected {expectedSize.Value} bytes, got {actualSize} bytes. " +
                    $"The cached file has been deleted. Set ForceRedownload=true or delete the cache directory.");
            }
        }

        // Streaming SHA256
        var actualSha256 = await ComputeSha256Async(filePath, ct);

        if (!string.Equals(actualSha256, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            File.Delete(filePath);
            throw new InvalidOperationException(
                $"SHA256 mismatch for {filePath}. " +
                $"Expected: {expectedSha256}, Got: {actualSha256}. " +
                $"The cached file has been deleted. Set ForceRedownload=true or delete the cache directory.");
        }
    }

    /// <summary>
    /// Checks if a file exists and has the expected SHA256 hash.
    /// Returns true if valid, false if missing or hash mismatch (does NOT delete on mismatch).
    /// </summary>
    public static async Task<bool> IsValidAsync(
        string filePath,
        string expectedSha256,
        long? expectedSize,
        CancellationToken ct)
    {
        if (!File.Exists(filePath))
            return false;

        if (expectedSize.HasValue)
        {
            var actualSize = new FileInfo(filePath).Length;
            if (actualSize != expectedSize.Value)
                return false;
        }

        var actualSha256 = await ComputeSha256Async(filePath, ct);
        return string.Equals(actualSha256, expectedSha256, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Computes SHA256 hash of a file using streaming (never loads full file into memory).</summary>
    public static async Task<string> ComputeSha256Async(string filePath, CancellationToken ct)
    {
        using var sha256 = SHA256.Create();
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, useAsync: true);

        var buffer = new byte[BufferSize];
        int bytesRead;
        while ((bytesRead = await stream.ReadAsync(buffer, ct)) > 0)
        {
            sha256.TransformBlock(buffer, 0, bytesRead, null, 0);
        }
        sha256.TransformFinalBlock([], 0, 0);

        return Convert.ToHexString(sha256.Hash!).ToLowerInvariant();
    }
}
