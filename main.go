// The go:build ignore tag prevents a "main redeclared" conflict
// when building the whole module. Run this file directly:
//
//	go run main.go

//go:build ignore

// main.go — Simple console-mode example using the DeepSeek model with ADK Runner.
//
// Usage:
//
//	DEEPSEEK_API_KEY=sk-... go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/abbas-zaidi/google-adk-go/deepseek"

	"github.com/joho/godotenv"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("DEEPSEEK_API_KEY environment variable not set")
	}

	// Create the DeepSeek model using the repo's adapter.
	llm, err := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey(apiKey),
	)
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	// Build an LLM agent with the model.
	a, err := llmagent.New(llmagent.Config{
		Name:        "deepseek-assistant",
		Model:       llm,
		Instruction: "You are a helpful AI assistant powered by DeepSeek.",
		Description: "An agent that uses DeepSeek language model.",
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	sessionSvc := session.InMemoryService()

	appRunner, err := runner.New(runner.Config{
		AppName:        "deepseek-app",
		Agent:          a,
		SessionService: sessionSvc,
	})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	// Create a session before running the agent.
	sessionResp, err := sessionSvc.Create(ctx, &session.CreateRequest{
		AppName:   "deepseek-app",
		UserID:    "user1",
		SessionID: "session1",
	})
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	message := "What is the capital of France?"
	fmt.Printf("User: %s\n", message)

	msg := genai.NewContentFromText(message, "user")
	for event, err := range appRunner.Run(ctx, "user1", sessionResp.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("runner error: %v", err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			var b strings.Builder
			for _, p := range event.Content.Parts {
				if p != nil && p.Text != "" {
					b.WriteString(p.Text)
				}
			}
			fmt.Printf("Agent: %s\n", b.String())
		}
	}
}
