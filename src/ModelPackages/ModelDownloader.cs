namespace ModelPackages;

/// <summary>
/// Downloads model artifacts via HTTP with streaming, authentication, retries, and progress logging.
/// </summary>
internal static class ModelDownloader
{
    private static readonly HttpClient SharedClient = CreateClient();
    private const int MaxRetries = 3;
    private const int BufferSize = 81920; // 80KB chunks

    /// <summary>Override in tests to eliminate retry delays.</summary>
    internal static Func<int, TimeSpan> RetryDelayFactory { get; set; } =
        attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt));

    private static HttpClient CreateClient()
    {
        var handler = new HttpClientHandler { AllowAutoRedirect = false };
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
        CancellationToken ct,
        HashSet<string>? allowedHosts = null)
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
                await DownloadCoreAsync(url, destinationPath, options, log, ct, allowedHosts);
                return; // Success
            }
            catch (HttpRequestException ex) when (attempt < MaxRetries && IsTransient(ex))
            {
                var delay = RetryDelayFactory(attempt);
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
        CancellationToken ct,
        HashSet<string>? allowedHosts)
    {
        var progress = options?.Progress;
        var fileName = Path.GetFileName(new Uri(url).AbsolutePath);

        try
        {
            long existingBytes = 0;
            if (File.Exists(destinationPath))
                existingBytes = new FileInfo(destinationPath).Length;

            // Allow one restart if Range request gets 416 or Content-Range mismatch
            for (int rangeAttempt = 0; rangeAttempt < 2; rangeAttempt++)
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
            if (existingBytes > 0)
            {
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingBytes, null);
                log($"Resuming download from byte {existingBytes}...");
            }

            log($"Downloading from {RedactUrl(url)}...");

            var response = await SharedClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);

            // Manual redirect handling with host validation
            const int maxRedirects = 10;
            int redirectCount = 0;
            var currentUri = new Uri(url);
            while (IsRedirectStatus(response.StatusCode) && redirectCount < maxRedirects)
            {
                var location = response.Headers.Location;
                if (location == null) break;

                var redirectUri = location.IsAbsoluteUri ? location : new Uri(currentUri, location);

                // Validate redirect target against allowed-host policy
                if (allowedHosts != null && allowedHosts.Count > 0 &&
                    !redirectUri.Scheme.Equals("file", StringComparison.OrdinalIgnoreCase) &&
                    !allowedHosts.Contains(redirectUri.Host))
                {
                    response.Dispose();
                    throw new InvalidOperationException(
                        $"Redirect to disallowed host '{redirectUri.Host}' blocked. " +
                        $"Allowed hosts: {string.Join(", ", allowedHosts)}. " +
                        $"Add '{redirectUri.Host}' to allowedHosts in model-sources.json if this redirect is expected.");
                }

                response.Dispose();
                using var redirectRequest = new HttpRequestMessage(HttpMethod.Get, redirectUri);
                // Preserve auth header only for same-host redirects
                if (request.Headers.Authorization != null &&
                    Uri.TryCreate(url, UriKind.Absolute, out var originalUri) &&
                    string.Equals(redirectUri.Host, originalUri.Host, StringComparison.OrdinalIgnoreCase))
                {
                    redirectRequest.Headers.Authorization = request.Headers.Authorization;
                }
                // Preserve Range header for resume support
                if (request.Headers.Range != null)
                    redirectRequest.Headers.Range = request.Headers.Range;

                log($"Following redirect to {RedactUrl(redirectUri.AbsoluteUri)}...");
                response = await SharedClient.SendAsync(redirectRequest, HttpCompletionOption.ResponseHeadersRead, ct);
                currentUri = redirectUri;
                redirectCount++;
            }

            if (IsRedirectStatus(response.StatusCode))
            {
                response.Dispose();
                throw new InvalidOperationException($"Too many redirects ({maxRedirects}) following {RedactUrl(url)}.");
            }

            using (response)
            {
            // Handle 416 (Range Not Satisfiable): delete partial, restart without Range
            if ((int)response.StatusCode == 416 && existingBytes > 0 && rangeAttempt == 0)
            {
                log("Range request failed (416). Deleting partial file and restarting from scratch.");
                if (File.Exists(destinationPath))
                    File.Delete(destinationPath);
                existingBytes = 0;
                continue;
            }

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

                throw new HttpRequestException(message, null, response.StatusCode);
            }

            // Determine if we're resuming or starting fresh
            bool resuming = existingBytes > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent;

            // Content-Range validation: ensure server is resuming from where we expect
            if (resuming)
            {
                var contentRange = response.Content.Headers.ContentRange;
                if (contentRange?.From != existingBytes && rangeAttempt == 0)
                {
                    log($"Content-Range mismatch (expected from={existingBytes}, got {contentRange}). Restarting from scratch.");
                    if (File.Exists(destinationPath))
                        File.Delete(destinationPath);
                    existingBytes = 0;
                    continue;
                }
            }

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
            return; // Success — downloaded within this range attempt
            } // using (response)
            } // for rangeAttempt
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception) when (progress != null)
        {
            progress.Report(new DownloadProgress(0, null, fileName, DownloadPhase.Failed));
            throw;
        }
    }

    private static bool IsRedirectStatus(System.Net.HttpStatusCode status)
    {
        var code = (int)status;
        return code is 301 or 302 or 303 or 307 or 308;
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
