# ModelPackages.Tool

CLI tool for prefetching, verifying, and managing model package artifacts.

## Install

```bash
dotnet tool install -g ModelPackages.Tool
```

## Commands

```bash
model-packages prefetch --manifest <path>    # Download and verify
model-packages verify   --manifest <path>    # Check cached model integrity
model-packages info     --manifest <path>    # Show source, cache path, metadata
model-packages clear-cache --manifest <path> # Remove cached model
```

See the [CLI reference](https://github.com/luisquintanilla/model-packages-prototype/blob/main/docs/cli-reference.md) for full details.
