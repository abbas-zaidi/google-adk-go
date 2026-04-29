package deepseek_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/abbas-zaidi/google-adk-go/deepseek"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestFullConversationWithToolCall(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		w.Header().Set("Content-Type", "application/json")

		switch callCount {
		case 1:
			// First call: model responds with a function call.
			json.NewEncoder(w).Encode(map[string]any{
				"id":    "resp-1",
				"model": "deepseek-chat",
				"choices": []map[string]any{{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Let me check the weather.",
						"tool_calls": []map[string]any{{
							"id":   "call_1",
							"type": "function",
							"function": map[string]any{
								"name":      "get_weather",
								"arguments": `{"city":"Delhi"}`,
							},
						}},
					},
				}},
				"usage": map[string]any{
					"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
				},
			})
		case 2:
			// Second call: model responds with final text after receiving tool result.
			json.NewEncoder(w).Encode(map[string]any{
				"id":    "resp-2",
				"model": "deepseek-chat",
				"choices": []map[string]any{{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "The weather in Delhi is sunny and 28°C.",
					},
				}},
				"usage": map[string]any{
					"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27,
				},
			})
		default:
			t.Fatalf("unexpected call count: %d", callCount)
		}
	}))
	defer srv.Close()

	m, _ := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey("test-key"),
		deepseek.WithBaseURL(srv.URL),
		deepseek.WithHTTPClient(srv.Client()),
	)

	// First call: user asks about weather.
	req1 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is the weather in Delhi?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are a helpful assistant.", "user"),
		},
	}

	var resp1 *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req1, false) {
		if err != nil {
			t.Fatalf("first call error: %v", err)
		}
		resp1 = resp
	}

	if resp1 == nil {
		t.Fatal("expected first response")
	}
	if resp1.Content == nil {
		t.Fatal("expected content in first response")
	}

	// Verify the response has a function call.
	hasFnCall := false
	for _, p := range resp1.Content.Parts {
		if p.FunctionCall != nil {
			hasFnCall = true
			break
		}
	}
	if !hasFnCall {
		t.Fatal("expected function call in first response")
	}
	if resp1.Partial {
		t.Error("first response should not be partial")
	}
	if !resp1.TurnComplete {
		t.Error("first response should have TurnComplete")
	}

	// Second call: include the function call + tool result in the conversation.
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is the weather in Delhi?", "user"),
			resp1.Content, // model response with function call
			{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "call_1",
						Name:     "get_weather",
						Response: map[string]any{"weather": "sunny and 28°C"},
					},
				}},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are a helpful assistant.", "user"),
		},
	}

	var resp2 *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req2, false) {
		if err != nil {
			t.Fatalf("second call error: %v", err)
		}
		resp2 = resp
	}

	if resp2 == nil {
		t.Fatal("expected second response")
	}
	if resp2.Content == nil {
		t.Fatal("expected content in second response")
	}
	if resp2.Partial {
		t.Error("second response should not be partial")
	}
	if !resp2.TurnComplete {
		t.Error("second response should have TurnComplete")
	}

	// Verify the final response has text content.
	hasText := false
	for _, p := range resp2.Content.Parts {
		if p.Text != "" {
			hasText = true
			break
		}
	}
	if !hasText {
		t.Fatal("expected text in final response")
	}
}
