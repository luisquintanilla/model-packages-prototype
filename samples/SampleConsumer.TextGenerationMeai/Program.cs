using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

Console.WriteLine("=== Provider-Agnostic Text Generation via IChatClient ===\n");
Console.WriteLine("Demonstrates wrapping any IChatClient as an ML.NET transform.");
Console.WriteLine("Swap DemoChatClient for OpenAI, Ollama, Azure OpenAI, etc.\n");

var mlContext = new MLContext();

// This is the ONLY part that changes per provider.
// For OpenAI:   new OpenAIClient(apiKey).GetChatClient("gpt-4o").AsIChatClient()
// For Ollama:   new OllamaChatClient(new Uri("http://localhost:11434"), "phi3")
IChatClient chatClient = new DemoChatClient();

var estimator = mlContext.Transforms.TextGeneration(chatClient, new TextGenerationOptions
{
    SystemPrompt = "You are a helpful assistant. Be concise.",
    MaxOutputTokens = 100,
    Temperature = 0.7f
});

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "Explain .NET in one sentence." },
    new TextData { Text = "What is the capital of France?" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
var transformer = estimator.Fit(dataView);
var transformed = transformer.Transform(dataView);

var results = mlContext.Data.CreateEnumerable<TextGenerationResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in results.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  Prompt:   \"{sampleData[idx].Text}\"");
    Console.WriteLine($"  Response: \"{item.GeneratedText}\"");
    Console.WriteLine();
}

Console.WriteLine("Note: Swap DemoChatClient for any IChatClient — pipeline code stays identical.");
Console.WriteLine("\nDone!");

public class TextData
{
    public string Text { get; set; } = "";
}

public class TextGenerationResult
{
    public string Text { get; set; } = "";
    public string GeneratedText { get; set; } = "";
}

/// <summary>
/// Demo IChatClient — replace with a real provider for actual generation.
/// </summary>
public class DemoChatClient : IChatClient
{
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var userMessage = messages.LastOrDefault(m => m.Role == ChatRole.User)?.Text ?? "";
        var reply = $"[Demo response to: {userMessage}]";
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, reply)));
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        throw new NotSupportedException("Streaming not supported in demo client.");
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() { }
}
