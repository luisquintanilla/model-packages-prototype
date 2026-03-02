using Xunit;

namespace ModelPackages.Tests;

public class AtomicWriteTests : IDisposable
{
    private readonly string _tempDir;

    public AtomicWriteTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "aw-test-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }

    [Fact]
    public async Task AtomicWrite_Success_FileExistsAtFinalPath()
    {
        var finalPath = Path.Combine(_tempDir, "model.bin");
        var content = new byte[] { 1, 2, 3, 4, 5 };

        await ModelCache.AtomicWriteAsync(finalPath, async tempPath =>
        {
            await File.WriteAllBytesAsync(tempPath, content);
        }, CancellationToken.None);

        Assert.True(File.Exists(finalPath));
        Assert.Equal(content, await File.ReadAllBytesAsync(finalPath));
    }

    [Fact]
    public async Task AtomicWrite_Success_NoTempFilesRemain()
    {
        var finalPath = Path.Combine(_tempDir, "clean.bin");

        await ModelCache.AtomicWriteAsync(finalPath, async tempPath =>
        {
            await File.WriteAllBytesAsync(tempPath, [42]);
        }, CancellationToken.None);

        var files = Directory.GetFiles(_tempDir);
        Assert.Single(files);
        Assert.Equal("clean.bin", Path.GetFileName(files[0]));
    }

    [Fact]
    public async Task AtomicWrite_CallbackThrows_TempFileCleaned()
    {
        var finalPath = Path.Combine(_tempDir, "failed.bin");

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await ModelCache.AtomicWriteAsync(finalPath, _ =>
            {
                throw new InvalidOperationException("download failed");
            }, CancellationToken.None);
        });

        Assert.False(File.Exists(finalPath));
        // No .partial files should remain
        var partials = Directory.GetFiles(_tempDir, "*.partial.*");
        Assert.Empty(partials);
    }

    [Fact]
    public async Task AtomicWrite_CreatesDirectoryIfNeeded()
    {
        var nestedDir = Path.Combine(_tempDir, "org", "model", "v1");
        var finalPath = Path.Combine(nestedDir, "model.bin");

        await ModelCache.AtomicWriteAsync(finalPath, async tempPath =>
        {
            await File.WriteAllBytesAsync(tempPath, [1, 2, 3]);
        }, CancellationToken.None);

        Assert.True(File.Exists(finalPath));
    }

    [Fact]
    public async Task AtomicWrite_OverwriteExisting_Succeeds()
    {
        var finalPath = Path.Combine(_tempDir, "overwrite.bin");
        await File.WriteAllBytesAsync(finalPath, [1]);

        await ModelCache.AtomicWriteAsync(finalPath, async tempPath =>
        {
            await File.WriteAllBytesAsync(tempPath, [2, 3, 4]);
        }, CancellationToken.None);

        Assert.Equal([2, 3, 4], await File.ReadAllBytesAsync(finalPath));
    }

    [Fact]
    public async Task AcquireLock_AndRelease_Works()
    {
        var lockTarget = Path.Combine(_tempDir, "locktest.bin");

        using (var lk = await ModelCache.AcquireLockAsync(lockTarget, CancellationToken.None))
        {
            Assert.True(File.Exists(lockTarget + ".lock"));
        }

        // Lock file is cleaned up after dispose
        Assert.False(File.Exists(lockTarget + ".lock"));
    }
}
