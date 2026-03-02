using System.Net;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace ModelPackages.Tests.Helpers;

/// <summary>
/// Lightweight in-process HTTP server for deterministic, offline download tests.
/// </summary>
internal sealed class MockModelServer : IAsyncDisposable
{
    private readonly WebApplication _app;
    private readonly Dictionary<string, FileEntry> _files = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<RequestRecord> _requests = [];

    public string BaseUrl { get; }
    public IReadOnlyList<RequestRecord> Requests => _requests;

    public MockModelServer()
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.ConfigureKestrel(k => k.Listen(IPAddress.Loopback, 0));
        _app = builder.Build();

        _app.MapGet("/{**path}", (HttpContext ctx) =>
        {
            var path = ctx.Request.Path.Value?.TrimStart('/') ?? "";
            _requests.Add(new RequestRecord(path, ctx.Request.Headers.Authorization.ToString()));

            if (!_files.TryGetValue(path, out var entry))
                return Results.NotFound($"File not found: {path}");

            if (entry.FailCount > 0)
            {
                entry.FailCount--;
                return Results.StatusCode(entry.FailStatusCode);
            }

            return Results.Bytes(entry.Content, "application/octet-stream");
        });

        _app.Start();
        var addresses = _app.Services.GetRequiredService<IServer>()
            .Features.Get<IServerAddressesFeature>()!;
        BaseUrl = addresses.Addresses.First();
    }

    /// <summary>Register a file to serve at the given path.</summary>
    public void AddFile(string path, byte[] content)
    {
        _files[path] = new FileEntry(content);
    }

    /// <summary>Register a file that fails the first N requests, then succeeds.</summary>
    public void AddFile(string path, byte[] content, int failCount, int failStatusCode = 500)
    {
        _files[path] = new FileEntry(content) { FailCount = failCount, FailStatusCode = failStatusCode };
    }

    public async ValueTask DisposeAsync()
    {
        await _app.DisposeAsync();
    }

    private sealed class FileEntry(byte[] content)
    {
        public byte[] Content { get; } = content;
        public int FailCount { get; set; }
        public int FailStatusCode { get; set; } = 500;
    }

    internal sealed record RequestRecord(string Path, string AuthorizationHeader);
}
