using Xunit;

namespace ModelPackages.Tests;

/// <summary>
/// Serializes test classes that mutate process-level environment variables.
/// xUnit runs tests in parallel by default; env var mutations are not parallel-safe.
/// </summary>
[CollectionDefinition("EnvVarTests")]
public class EnvVarTestCollection : ICollectionFixture<EnvVarTestCollection.Marker>
{
    public class Marker { }
}
