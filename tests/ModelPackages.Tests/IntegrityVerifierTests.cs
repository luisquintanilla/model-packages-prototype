using System.Security.Cryptography;
using System.Text;
using Xunit;

namespace ModelPackages.Tests;

public class IntegrityVerifierTests : IDisposable
{
    private readonly string _tempDir;

    public IntegrityVerifierTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "iv-test-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }

    private string CreateTestFile(string content = "test model content")
    {
        var path = Path.Combine(_tempDir, Guid.NewGuid().ToString("N") + ".bin");
        File.WriteAllText(path, content);
        return path;
    }

    private static string ComputeSha256(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        return Convert.ToHexString(sha256.ComputeHash(stream)).ToLowerInvariant();
    }

    [Fact]
    public async Task VerifyAsync_ValidFile_Passes()
    {
        var filePath = CreateTestFile();
        var expectedHash = ComputeSha256(filePath);
        var expectedSize = new FileInfo(filePath).Length;

        // Should not throw
        await IntegrityVerifier.VerifyAsync(filePath, expectedHash, expectedSize, CancellationToken.None);
    }

    [Fact]
    public async Task VerifyAsync_WrongSha256_ThrowsAndDeletesFile()
    {
        var filePath = CreateTestFile();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            IntegrityVerifier.VerifyAsync(filePath, "0000000000000000000000000000000000000000000000000000000000000000", null, CancellationToken.None));

        Assert.Contains("SHA256 mismatch", ex.Message);
        Assert.False(File.Exists(filePath));
    }

    [Fact]
    public async Task VerifyAsync_WrongSize_ThrowsAndDeletesFile()
    {
        var filePath = CreateTestFile();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            IntegrityVerifier.VerifyAsync(filePath, "unused", 999999L, CancellationToken.None));

        Assert.Contains("Size mismatch", ex.Message);
        Assert.False(File.Exists(filePath));
    }

    [Fact]
    public async Task VerifyAsync_SizeZero_SkipsSizeCheck()
    {
        var filePath = CreateTestFile("some content");
        var expectedHash = ComputeSha256(filePath);

        // size=0 should be treated as "unknown" — skip size validation
        await IntegrityVerifier.VerifyAsync(filePath, expectedHash, 0L, CancellationToken.None);

        Assert.True(File.Exists(filePath));
    }

    [Fact]
    public async Task VerifyAsync_EmptySha256_SkipsWithWarning()
    {
        var filePath = CreateTestFile();
        var warnings = new List<string>();

        await IntegrityVerifier.VerifyAsync(filePath, "", null, CancellationToken.None,
            log: msg => warnings.Add(msg));

        Assert.True(File.Exists(filePath));
        Assert.Contains(warnings, w => w.Contains("No SHA256 hash"));
    }

    [Fact]
    public async Task VerifyAsync_NullSha256_SkipsWithWarning()
    {
        var filePath = CreateTestFile();
        var warnings = new List<string>();

        await IntegrityVerifier.VerifyAsync(filePath, null!, null, CancellationToken.None,
            log: msg => warnings.Add(msg));

        Assert.True(File.Exists(filePath));
        Assert.Contains(warnings, w => w.Contains("No SHA256 hash"));
    }

    [Fact]
    public async Task VerifyAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            IntegrityVerifier.VerifyAsync(
                Path.Combine(_tempDir, "nonexistent.bin"),
                "abc",
                null,
                CancellationToken.None));
    }

    [Fact]
    public async Task IsValidAsync_ValidFile_ReturnsTrue()
    {
        var filePath = CreateTestFile();
        var expectedHash = ComputeSha256(filePath);
        var expectedSize = new FileInfo(filePath).Length;

        var result = await IntegrityVerifier.IsValidAsync(filePath, expectedHash, expectedSize, CancellationToken.None);

        Assert.True(result);
    }

    [Fact]
    public async Task IsValidAsync_MissingFile_ReturnsFalse()
    {
        var result = await IntegrityVerifier.IsValidAsync(
            Path.Combine(_tempDir, "nonexistent.bin"),
            "abc",
            null,
            CancellationToken.None);

        Assert.False(result);
    }

    [Fact]
    public async Task IsValidAsync_EmptySha256_ReturnsFalse()
    {
        var filePath = CreateTestFile();

        var result = await IntegrityVerifier.IsValidAsync(filePath, "", null, CancellationToken.None);

        Assert.False(result);
    }

    [Fact]
    public async Task ComputeSha256Async_EmptyFile_ReturnsValidHash()
    {
        var filePath = Path.Combine(_tempDir, "empty.bin");
        File.WriteAllBytes(filePath, []);

        var hash = await IntegrityVerifier.ComputeSha256Async(filePath, CancellationToken.None);

        // SHA256 of empty content is well-known
        Assert.Equal("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", hash);
    }
}
