namespace ModelPackages;

/// <summary>
/// Downloads model artifacts via HTTP with streaming, authentication, retries, and progress logging.
/// </summary>
internal static class ModelDownloader
{
    private static readonly HttpClient SharedClient = CreateClient();
    private const int MaxRetries = 3;
    private const int BufferSize = 81920; // 80KB chunks

    private static HttpClient CreateClient()
    {
        var handler = new HttpClientHandler { AllowAutoRedirect = true };
        var client = new HttpClient(handler);
        client.DefaultRequestHeaders.UserAgent.ParseAdd("ModelPackages/1.0");
        return client;
    }

    /// <summary>
    /// Downloads a file from the given URL to the destination path using streaming.
    /// Supports file:// URIs (local copy), HuggingFace bearer token auth, and retry with exponential backoff.
    /// </summary>
    public static async Task DownloadAsync(
        string url,
        string destinationPath,
        ModelOptions? options,
        CancellationToken ct)
    {
        var log = options?.Logger ?? (_ => { });

        // Handle file:// URIs as local file copy
        if (url.StartsWith("file://", StringComparison.OrdinalIgnoreCase))
        {
            var sourcePath = new Uri(url).LocalPath;
            log($"Copying from local path: {sourcePath}");
            using var src = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, useAsync: true);
            using var dst = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, BufferSize, useAsync: true);
            await src.CopyToAsync(dst, ct);
            log($"Copy complete: {new FileInfo(destinationPath).Length / 1024 / 1024} MB");
            return;
        }

        for (int attempt = 1; attempt <= MaxRetries; attempt++)
        {
            try
            {
                await DownloadCoreAsync(url, destinationPath, options, log, ct);
                return; // Success
            }
            catch (HttpRequestException ex) when (attempt < MaxRetries && IsTransient(ex))
            {
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                log($"Download attempt {attempt} failed ({ex.Message}). Retrying in {delay.TotalSeconds}s...");
                await Task.Delay(delay, ct);
            }
        }
    }

    private static async Task DownloadCoreAsync(
        string url,
        string destinationPath,
        ModelOptions? options,
        Action<string> log,
        CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);

        // HuggingFace auth: bearer token from options or HF_TOKEN env
        var token = options?.HuggingFaceToken
            ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        }

        log($"Downloading from {RedactUrl(url)}...");

        using var response = await SharedClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);

        if (!response.IsSuccessStatusCode)
        {
            var statusCode = (int)response.StatusCode;
            var message = statusCode switch
            {
                401 or 403 => $"HTTP {statusCode}: Authentication failed. Set HF_TOKEN for private HF repos or override MODELPACKAGES_SOURCE.",
                404 => $"HTTP {statusCode}: Model file not found at {RedactUrl(url)}. Check the manifest source configuration.",
                _ => $"HTTP {statusCode}: Download failed from {RedactUrl(url)}."
            };
            throw new HttpRequestException(message, null, response.StatusCode);
        }

        var totalBytes = response.Content.Headers.ContentLength;
        log($"Content-Length: {(totalBytes.HasValue ? $"{totalBytes.Value / 1024 / 1024} MB" : "unknown")}");

        using var contentStream = await response.Content.ReadAsStreamAsync(ct);
        using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, BufferSize, useAsync: true);

        var buffer = new byte[BufferSize];
        long totalRead = 0;
        int bytesRead;
        var lastReport = DateTimeOffset.UtcNow;

        while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
            totalRead += bytesRead;

            // Report progress every 5 seconds
            if (DateTimeOffset.UtcNow - lastReport > TimeSpan.FromSeconds(5))
            {
                if (totalBytes.HasValue)
                    log($"Progress: {totalRead / 1024 / 1024} MB / {totalBytes.Value / 1024 / 1024} MB ({100.0 * totalRead / totalBytes.Value:F1}%)");
                else
                    log($"Progress: {totalRead / 1024 / 1024} MB downloaded");
                lastReport = DateTimeOffset.UtcNow;
            }
        }

        log($"Download complete: {totalRead / 1024 / 1024} MB");
    }

    private static bool IsTransient(HttpRequestException ex)
    {
        // Retry on 429 (rate limit) and 5xx (server errors), not on 401/403/404
        if (ex.StatusCode.HasValue)
        {
            var code = (int)ex.StatusCode.Value;
            return code == 429 || code >= 500;
        }
        // Retry on connection errors (no status code)
        return true;
    }

    private static string RedactUrl(string url)
    {
        // Remove query strings that might contain tokens
        var idx = url.IndexOf('?');
        return idx >= 0 ? url[..idx] + "?[REDACTED]" : url;
    }
}
