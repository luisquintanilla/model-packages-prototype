namespace ModelPackages;

/// <summary>Structured progress data reported during model download and verification.</summary>
public readonly record struct DownloadProgress(
    long BytesTransferred,
    long? TotalBytes,
    string FileName,
    DownloadPhase Phase);

/// <summary>Phases of the model download lifecycle.</summary>
public enum DownloadPhase
{
    /// <summary>Determining download URL from source configuration.</summary>
    Resolving,

    /// <summary>Actively downloading bytes from the remote source.</summary>
    Downloading,

    /// <summary>Computing SHA256 hash to verify integrity.</summary>
    Verifying,

    /// <summary>Download and verification completed successfully.</summary>
    Completed,

    /// <summary>An error occurred during download or verification.</summary>
    Failed
}
