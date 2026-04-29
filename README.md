# google-adk-go

DeepSeek model adapter for [Google ADK (Agent Development Kit) for Go][adk-go] — use DeepSeek models as the LLM backend in ADK agents with full tool-calling support.

[adk-go]: https://pkg.go.dev/google.golang.org/adk

## Quick start

```bash
git clone https://github.com/abbas-zaidi/google-adk-go.git
cd google-adk-go

# Set your DeepSeek API key
export DEEPSEEK_API_KEY=sk-...

# Run the agent (web UI + REST API)
go run agent.go

# Or run the console example
go run main.go
```

## How it works

The `deepseek/` package implements `model.LLM`, the core LLM interface in ADK. It translates between ADK's `genai.Content` types and DeepSeek's [OpenAI-compatible chat completion API][deepseek-api], handling:

- **Messages** — system instructions, user/assistant/tool roles
- **Tool calls** — function declarations → OpenAI tools, ADK function calls/responses → DeepSeek tool messages
- **Streaming** — SSE parsing (disabled by default in favor of non-streaming for reliable tool execution)
- **Usage metadata** — token counts mapped to ADK's usage struct

```go
import "github.com/abbas-zaidi/google-adk-go/deepseek"

model, err := deepseek.New(
    deepseek.ModelDeepSeekChat,
    deepseek.WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
)
```

See [`agent.go`](agent.go) for a complete example with a `get_weather` function tool.

[deepseek-api]: https://api-docs.deepseek.com/

## Available models

| Constant | API model |
|---|---|
| `ModelDeepSeekChat` | `deepseek-chat` |
| `ModelDeepSeekReasoner` | `deepseek-reasoner` |

## Run tests

```bash
go test ./deepseek/ -v
```

## Dependencies

- [google.golang.org/adk](https://pkg.go.dev/google.golang.org/adk) v1.0.0
- [google.golang.org/genai](https://pkg.go.dev/google.golang.org/genai)
