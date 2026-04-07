package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/voocel/litellm"
)

func main() {

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("DEEPSEEK_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("deepseek", litellm.ProviderConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("DeepSeek Examples - From Basic to Advanced")
	fmt.Println("=========================================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example (DeepSeek Chat)")
	fmt.Println("-------------------------------------")
	basicChat(client)
}

func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "You are a helpful AI assistant.",
			},
			{
				Role:    "user",
				Content: "Explain what DeepSeek is in simple terms.",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.7),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Basic chat failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)

	// Calculate cost (lazy loads pricing data on first call)
	if cost, err := litellm.CalculateCostForResponse(response); err == nil {
		fmt.Printf("Cost: $%.6f (input: $%.6f, output: $%.6f)\n", cost.TotalCost, cost.InputCost, cost.OutputCost)
	} else {
		fmt.Printf("Cost calculation: %v\n", err)
	}
}
