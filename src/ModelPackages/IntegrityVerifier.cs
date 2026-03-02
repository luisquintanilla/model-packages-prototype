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
        CancellationToken ct,
        Action<string>? log = null)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found at: {filePath}");

        // Optional size check (fast, before hashing) — skip when size is 0 or null (unknown)
        if (expectedSize.HasValue && expectedSize.Value > 0)
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

        // Skip SHA256 verification when hash is empty/missing (development workflow)
        if (string.IsNullOrEmpty(expectedSha256))
        {
            log?.Invoke($"Warning: No SHA256 hash in manifest for {Path.GetFileName(filePath)}. Skipping integrity verification. Populate the hash before publishing.");
            return;
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

        if (expectedSize.HasValue && expectedSize.Value > 0)
        {
            var actualSize = new FileInfo(filePath).Length;
            if (actualSize != expectedSize.Value)
                return false;
        }

        // If SHA256 is empty/missing, we can't validate — treat as invalid (forces re-download)
        if (string.IsNullOrEmpty(expectedSha256))
            return false;

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

    // ── Sidecar fast-path ──────────────────────────────────────────

    private static string SidecarPath(string filePath) => filePath + ".sha256";

    /// <summary>
    /// Fast validation using a sidecar .sha256 file. Checks size + mtime instead of re-hashing.
    /// Returns true if sidecar exists, metadata matches, and stored hash matches expected.
    /// </summary>
    public static bool QuickValidate(string filePath, string expectedSha256, long? expectedSize)
    {
        if (string.IsNullOrEmpty(expectedSha256))
            return false;

        var sidecar = SidecarPath(filePath);
        if (!File.Exists(sidecar) || !File.Exists(filePath))
            return false;

        try
        {
            var lines = File.ReadAllLines(sidecar);
            string? storedHash = null;
            long? storedSize = null;
            DateTime? storedMtime = null;

            foreach (var line in lines)
            {
                if (line.StartsWith("sha256=", StringComparison.Ordinal))
                    storedHash = line["sha256=".Length..];
                else if (line.StartsWith("size=", StringComparison.Ordinal) &&
                         long.TryParse(line["size=".Length..], out var sz))
                    storedSize = sz;
                else if (line.StartsWith("mtime=", StringComparison.Ordinal) &&
                         DateTime.TryParse(line["mtime=".Length..], null,
                             System.Globalization.DateTimeStyles.RoundtripKind, out var mt))
                    storedMtime = mt;
            }

            if (storedHash is null || storedSize is null || storedMtime is null)
                return false;

            var info = new FileInfo(filePath);
            if (info.Length != storedSize.Value)
                return false;
            if (info.LastWriteTimeUtc != storedMtime.Value)
                return false;

            // Sidecar metadata matches — trust the stored hash
            if (!string.Equals(storedHash, expectedSha256, StringComparison.OrdinalIgnoreCase))
                return false;

            // Size from manifest (if known) should also agree
            if (expectedSize.HasValue && expectedSize.Value > 0 && info.Length != expectedSize.Value)
                return false;

            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Writes a sidecar .sha256 file next to the model file with hash, size, and mtime.
    /// </summary>
    public static void WriteSidecar(string filePath, string sha256)
    {
        var info = new FileInfo(filePath);
        var content = $"sha256={sha256}\nsize={info.Length}\nmtime={info.LastWriteTimeUtc:O}\n";
        File.WriteAllText(SidecarPath(filePath), content);
    }

    /// <summary>Deletes the sidecar file if it exists.</summary>
    public static void DeleteSidecar(string filePath)
    {
        var sidecar = SidecarPath(filePath);
        if (File.Exists(sidecar))
            File.Delete(sidecar);
    }
}
