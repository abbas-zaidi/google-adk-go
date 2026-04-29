# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo demonstrates integrating a **DeepSeek language model** with **Google ADK (Agent Development Kit) for Go**. It provides a custom `model.LLM` implementation (`deepseek` package) that translates between ADK's internal `genai.Content` / `model.LLMRequest` / `model.LLMResponse` types and the OpenAI-compatible chat completion format used by the DeepSeek API.

## Build / run / test

```bash
# Run the ADK agent with web UI + REST API (primary entry point)
DEEPSEEK_API_KEY=sk-... go run agent.go

# Run the simpler console-mode example (hardcoded test message)
DEEPSEEK_API_KEY=sk-... go run main.go

# Run deepseek package tests
cd deepseek && go test -v ./...
```

## Architecture

### Module structure

The repo has **two separate Go modules** with no `go.work` file:

- **Root module** (`github.com/abbas-zaidi/google-adk-go`) — the application entry points (`agent.go`, `main.go`)
- **`deepseek/` module** (`module deepseek`) — the reusable model adapter package

`agent.go` and tests import the deepseek package as `"google-adk-go/deepseek"`. This is a relative-style import that relies on the directory layout; there is no `replace` directive in the root `go.mod`.

### deepseek package (`deepseek/deepseek.go`)

The core adapter. `Model` implements `model.LLM` from ADK:

- **`New(modelName, ...opts)`** — constructor with functional options (`WithAPIKey`, `WithBaseURL`, `WithHTTPClient`). Falls back to `DEEPSEEK_API_KEY` env var.
- **`GenerateContent(ctx, req, stream)`** — returns `iter.Seq2[*model.LLMResponse, error]` (Go 1.23+ iterator pattern). Converts ADK request → OpenAI-style JSON → POST to `{baseURL}/chat/completions` → converts response back to ADK types. Handles both streaming (SSE) and non-streaming responses.
- **`contentsToMessages`** — converts `genai.Content` to OpenAI chat messages, injecting system instruction as the first message if present, and ensuring the final message is from the user.
- **`buildTools`** — converts ADK tool definitions to OpenAI function-calling format.
- **`buildLLMResponse`** — converts DeepSeek's assistant message (text + tool calls) into ADK's `model.LLMResponse` with `genai.Content` parts.

Key constants: `ModelDeepSeekChat = "deepseek-chat"`, `ModelDeepSeekReasoner = "deepseek-reasoner"`.

### agent.go (primary app)

Uses ADK's **launcher** package to run an agent with a built-in web UI and REST API. Defines a demo `get_weather` function tool via `functiontool.New`. The `NewRootAgent` factory wires the DeepSeek model, weather tool, and system instruction together into an `llmagent`.

### main.go (simpler/older example)

Uses the `github.com/go-deepseek/deepseek` SDK directly (not the custom `deepseek` package) and ADK's `runner.Runner` for a one-shot console interaction. Implements an older version of the `model.LLM` interface where `GenerateContent` returns `(*model.LLMResponse, error)` — this interface signature changed in ADK v1.1.0.

### Testing pattern (`deepseek/deepseek_test.go`)

Uses `httptest.NewServer` to mock the DeepSeek API. The functional options pattern (`WithBaseURL`, `WithHTTPClient`) allows pointing the model at the test server. Tests use `t.Setenv` to control environment state.

## Environment

- **`DEEPSEEK_API_KEY`** (required) — DeepSeek API key, read by both `deepseek.New()` and the older `main.go`
