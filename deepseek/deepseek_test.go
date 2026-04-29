package deepseek_test

import (
	"context"

	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/abbas-zaidi/google-adk-go/deepseek"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// fakeResponse returns an HTTP handler that responds with a canned DeepSeek body.
func fakeResponse(body string, status int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}
}

const sampleChatResponse = `{
  "id": "test-id",
  "model": "deepseek-chat",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"}
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}`

func TestGenerateContent_NonStreaming(t *testing.T) {
	srv := httptest.NewServer(fakeResponse(sampleChatResponse, http.StatusOK))
	defer srv.Close()

	m, err := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey("test-key"),
		deepseek.WithBaseURL(srv.URL),
		deepseek.WithHTTPClient(srv.Client()),
	)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
	}

	var got []*model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}
		got = append(got, resp)
	}

	if len(got) != 1 {
		t.Fatalf("expected 1 response, got %d", len(got))
	}
	resp := got[0]
	if resp.Content == nil {
		t.Fatal("expected non-nil Content")
	}
	text := ""
	for _, p := range resp.Content.Parts {
		if p != nil && p.Text != "" {
			text += p.Text
		}
	}
	if !strings.Contains(text, "Hello") {
		t.Errorf("expected response to contain 'Hello', got: %q", text)
	}
	if resp.UsageMetadata == nil {
		t.Error("expected non-nil UsageMetadata")
	} else if resp.UsageMetadata.TotalTokenCount != 18 {
		t.Errorf("expected total tokens 18, got %d", resp.UsageMetadata.TotalTokenCount)
	}
}

func TestGenerateContent_APIError(t *testing.T) {
	srv := httptest.NewServer(fakeResponse(`{"error":"unauthorized"}`, http.StatusUnauthorized))
	defer srv.Close()

	m, _ := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey("bad-key"),
		deepseek.WithBaseURL(srv.URL),
		deepseek.WithHTTPClient(srv.Client()),
	)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hi", "user"),
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			if !strings.Contains(err.Error(), "401") {
				t.Errorf("expected 401 error, got: %v", err)
			}
			return
		}
	}
	t.Error("expected an error but got none")
}

func TestGenerateContent_WithSystemInstruction(t *testing.T) {
	var capturedBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(sampleChatResponse))
	}))
	defer srv.Close()

	m, _ := deepseek.New(
		deepseek.ModelDeepSeekChat,
		deepseek.WithAPIKey("test-key"),
		deepseek.WithBaseURL(srv.URL),
		deepseek.WithHTTPClient(srv.Client()),
	)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Who are you?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a pirate. Speak like one."}},
			},
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	var sent map[string]any
	if err := json.Unmarshal(capturedBody, &sent); err != nil {
		t.Fatalf("could not parse sent body: %v", err)
	}
	msgs := sent["messages"].([]any)
	first := msgs[0].(map[string]any)
	if first["role"] != "system" {
		t.Errorf("expected first message role to be 'system', got %q", first["role"])
	}
	if !strings.Contains(first["content"].(string), "pirate") {
		t.Errorf("expected system instruction in first message")
	}
}

func TestNew_MissingAPIKey(t *testing.T) {
	// Ensure env var is clear for this test.
	t.Setenv("DEEPSEEK_API_KEY", "")
	_, err := deepseek.New(deepseek.ModelDeepSeekChat)
	if err == nil {
		t.Error("expected error when API key is missing")
	}
}
