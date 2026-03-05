using System.Security.Cryptography;
using ModelPackages.Tests.Helpers;
using Xunit;

namespace ModelPackages.Tests;

public class DownloaderTests : IAsyncLifetime
{
    private MockModelServer _server = null!;
    private string _tempDir = null!;
    private Func<int, TimeSpan>? _prevRetryDelay;

    public async Task InitializeAsync()
    {
        _server = new MockModelServer();
        _tempDir = Path.Combine(Path.GetTempPath(), "dl-test-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_tempDir);
        _prevRetryDelay = ModelDownloader.RetryDelayFactory;
        ModelDownloader.RetryDelayFactory = _ => TimeSpan.Zero;
        await Task.CompletedTask;
    }

    public async Task DisposeAsync()
    {
        ModelDownloader.RetryDelayFactory = _prevRetryDelay!;
        await _server.DisposeAsync();
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }

    private string TempFile(string name = "downloaded.bin") => Path.Combine(_tempDir, name);

    [Fact]
    public async Task Download_Success_FileMatchesContent()
    {
        var content = new byte[1024];
        Random.Shared.NextBytes(content);
        _server.AddFile("models/test.bin", content);

        var dest = TempFile();
        await ModelDownloader.DownloadAsync(
            $"{_server.BaseUrl}/models/test.bin",
            dest,
            options: null,
            CancellationToken.None);

        Assert.True(File.Exists(dest));
        Assert.Equal(content, await File.ReadAllBytesAsync(dest));
    }

    [Fact]
    public async Task Download_Http404_ThrowsWithMessage()
    {
        var dest = TempFile();

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            ModelDownloader.DownloadAsync(
                $"{_server.BaseUrl}/nonexistent",
                dest,
                options: null,
                CancellationToken.None));

        Assert.Contains("404", ex.Message);
    }

    [Fact]
    public async Task Download_Http500ThenSuccess_RetriesAndSucceeds()
    {
        var content = new byte[] { 1, 2, 3, 4, 5 };
        _server.AddFile("models/retry.bin", content, failCount: 1, failStatusCode: 500);

        var dest = TempFile();
        await ModelDownloader.DownloadAsync(
            $"{_server.BaseUrl}/models/retry.bin",
            dest,
            options: null,
            CancellationToken.None);

        Assert.Equal(content, await File.ReadAllBytesAsync(dest));
    }

    [Fact]
    public async Task Download_FileUrl_CopiesLocalFile()
    {
        var sourceFile = Path.Combine(_tempDir, "source-model.bin");
        var content = new byte[] { 10, 20, 30, 40, 50 };
        await File.WriteAllBytesAsync(sourceFile, content);

        var dest = TempFile("copied.bin");
        var fileUrl = new Uri(sourceFile).AbsoluteUri;

        await ModelDownloader.DownloadAsync(fileUrl, dest, options: null, CancellationToken.None);

        Assert.Equal(content, await File.ReadAllBytesAsync(dest));
    }

    [Fact]
    public async Task Download_HfToken_SentAsBearerAuth()
    {
        var content = new byte[] { 1 };
        _server.AddFile("models/private.bin", content);

        var dest = TempFile();
        var options = new ModelOptions { HuggingFaceToken = "hf_test_token_123" };

        await ModelDownloader.DownloadAsync(
            $"{_server.BaseUrl}/models/private.bin",
            dest,
            options,
            CancellationToken.None);

        var req = _server.Requests.Last();
        Assert.Contains("Bearer hf_test_token_123", req.AuthorizationHeader);
    }

    [Fact]
    public async Task Download_LoggerCallback_ReceivesMessages()
    {
        var content = new byte[1024];
        _server.AddFile("models/logged.bin", content);

        var messages = new List<string>();
        var options = new ModelOptions { Logger = msg => messages.Add(msg) };

        var dest = TempFile();
        await ModelDownloader.DownloadAsync(
            $"{_server.BaseUrl}/models/logged.bin",
            dest,
            options,
            CancellationToken.None);

        Assert.Contains(messages, m => m.Contains("Downloading"));
        Assert.Contains(messages, m => m.Contains("complete"));
    }

    [Fact]
    public async Task Download_Http429ThenSuccess_RetriesAndSucceeds()
    {
        var content = new byte[] { 42 };
        _server.AddFile("models/rate-limited.bin", content, failCount: 1, failStatusCode: 429);

        var dest = TempFile();
        await ModelDownloader.DownloadAsync(
            $"{_server.BaseUrl}/models/rate-limited.bin",
            dest,
            options: null,
            CancellationToken.None);

        Assert.Equal(content, await File.ReadAllBytesAsync(dest));
    }
}
