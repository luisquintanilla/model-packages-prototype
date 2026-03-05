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
    /// Supports file:// URIs (local copy), HuggingFace bearer token auth, retry with exponential backoff,
    /// and resumable downloads via HTTP Range headers.
    /// </summary>
    public static async Task DownloadAsync(
        string url,
        string destinationPath,
        ModelOptions? options,
        CancellationToken ct)
    {
        var log = options?.Logger ?? (_ => { });

        // Handle file:// URIs as local file copy (no resume support)
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
                // Partial file is preserved for resume on next attempt
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
        var progress = options?.Progress;
        var fileName = Path.GetFileName(destinationPath);

        try
        {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);

        // HuggingFace auth: bearer token from options or HF_TOKEN env
        var token = options?.HuggingFaceToken
            ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        }

        // Resume support: if partial file exists, request remaining bytes
        long existingBytes = 0;
        if (File.Exists(destinationPath))
        {
            existingBytes = new FileInfo(destinationPath).Length;
            if (existingBytes > 0)
            {
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingBytes, null);
                log($"Resuming download from byte {existingBytes}...");
            }
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
                416 => $"HTTP {statusCode}: Range not satisfiable for {RedactUrl(url)}. Partial file may be corrupt.",
                _ => $"HTTP {statusCode}: Download failed from {RedactUrl(url)}."
            };

            // If Range request fails (416), delete partial and the caller's retry will start fresh
            if (statusCode == 416 && File.Exists(destinationPath))
            {
                File.Delete(destinationPath);
            }

            progress?.Report(new DownloadProgress(0, null, fileName, DownloadPhase.Failed));
            throw new HttpRequestException(message, null, response.StatusCode);
        }

        // Determine if we're resuming or starting fresh
        bool resuming = existingBytes > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent;
        if (existingBytes > 0 && !resuming)
        {
            // Server doesn't support Range — restart from scratch
            log("Server does not support Range requests. Restarting download from scratch.");
            existingBytes = 0;
        }

        var totalBytes = response.Content.Headers.ContentLength;
        var totalExpected = resuming ? existingBytes + totalBytes : totalBytes;
        log($"Content-Length: {(totalBytes.HasValue ? $"{totalBytes.Value / 1024 / 1024} MB" : "unknown")}" +
            (resuming ? $" (resuming from {existingBytes / 1024 / 1024} MB)" : ""));

        using var contentStream = await response.Content.ReadAsStreamAsync(ct);
        // Append if resuming, create new otherwise
        using var fileStream = new FileStream(
            destinationPath,
            resuming ? FileMode.Append : FileMode.Create,
            FileAccess.Write, FileShare.None, BufferSize, useAsync: true);

        var buffer = new byte[BufferSize];
        long totalRead = existingBytes;
        int bytesRead;
        var lastReport = DateTimeOffset.UtcNow;
        var lastLogReport = DateTimeOffset.UtcNow;
        var minReportInterval = TimeSpan.FromMilliseconds(100);

        progress?.Report(new DownloadProgress(0, totalBytes, fileName, DownloadPhase.Downloading));

        while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
            totalRead += bytesRead;

            var now = DateTimeOffset.UtcNow;
            if (now - lastReport > minReportInterval)
            {
                progress?.Report(new DownloadProgress(totalRead, totalBytes, fileName, DownloadPhase.Downloading));
                lastReport = now;
            }

            // Log text progress every 5 seconds (separate timestamp from IProgress throttling)
            if (now - lastLogReport > TimeSpan.FromSeconds(5))
            {
                if (totalExpected.HasValue)
                    log($"Progress: {totalRead / 1024 / 1024} MB / {totalExpected.Value / 1024 / 1024} MB ({100.0 * totalRead / totalExpected.Value:F1}%)");
                else
                    log($"Progress: {totalRead / 1024 / 1024} MB downloaded");
                lastLogReport = now;
            }
        }

        // Don't report Completed here — let ModelPackage report it after verification succeeds
        log($"Download complete: {totalRead / 1024 / 1024} MB");
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception) when (progress != null)
        {
            progress.Report(new DownloadProgress(0, null, fileName, DownloadPhase.Failed));
            throw;
        }
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
