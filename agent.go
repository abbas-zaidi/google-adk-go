// agent.go — Example ADK Go agent using the custom DeepSeek model.
//
// Usage:
//
//	DEEPSEEK_API_KEY=sk-... go run agent.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/abbas-zaidi/google-adk-go/deepseek"
	"github.com/joho/godotenv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// ── Tool definitions ─────────────────────────────────────────────────────────

// WeatherArgs are the inputs for the get_weather tool.
type WeatherArgs struct {
	City string `json:"city" jsonschema:"City name to fetch weather for"`
}

// WeatherResult is returned by the tool.
type WeatherResult struct {
	Weather string `json:"weather"`
}

// GetWeather is a simple demo tool that returns mock weather data.
func GetWeather(_ tool.Context, args WeatherArgs) (WeatherResult, error) {
	return WeatherResult{
		Weather: fmt.Sprintf("It's sunny and 28°C in %s.", args.City),
	}, nil
}

// ── Agent factory ─────────────────────────────────────────────────────────────

// NewRootAgent constructs the ADK agent wired to the DeepSeek model.
func NewRootAgent(ctx context.Context) (agent.Agent, error) {
	// 1. Create the DeepSeek model.
	//    API key is read from DEEPSEEK_API_KEY by default.
	//    Swap to ModelDeepSeekReasoner for chain-of-thought tasks.
	dsModel, err := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
		// Uncomment to point at a local proxy instead:
		// deepseek.WithBaseURL("http://localhost:8080/v1"),
	)
	if err != nil {
		return nil, fmt.Errorf("creating deepseek model: %w", err)
	}

	// 2. Build function tools using ADK's functiontool package.
	weatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Returns the current weather for a given city.",
	}, GetWeather)
	if err != nil {
		return nil, fmt.Errorf("creating weather tool: %w", err)
	}

	// 3. Build the LLM agent, passing dsModel directly.
	//    ADK accepts any value implementing model.LLM.
	a, err := llmagent.New(llmagent.Config{
		Name:        "deepseek-assistant",
		Description: "A helpful assistant powered by DeepSeek.",
		Instruction: `You are a helpful assistant. When asked about the weather,
use the get_weather tool. Be concise and friendly.`,
		Model: dsModel, // ← our custom model.LLM implementation
		Tools: []tool.Tool{weatherTool},
	})
	if err != nil {
		return nil, fmt.Errorf("creating llm agent: %w", err)
	}
	return a, nil
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	ctx := context.Background()

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	a, err := NewRootAgent(ctx)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	// Launch the ADK web UI + REST API.
	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}, os.Args[1:]); err != nil {
		log.Fatalf("launcher error: %v", err)
	}
}
