// Package deepseek provides a custom model implementation for Google ADK Go
// that routes LLM calls to the DeepSeek API instead of Gemini.
//
// The DeepSeek API is OpenAI-compatible, so we translate between ADK's
// internal genai types and the OpenAI chat completion format.
package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	// DefaultBaseURL is the DeepSeek API base URL.
	DefaultBaseURL = "https://api.deepseek.com/v1"

	// ModelDeepSeekChat is the general-purpose chat model.
	ModelDeepSeekChat = "deepseek-chat"

	// ModelDeepSeekReasoner is the reasoning/CoT model (deepseek-r1).
	ModelDeepSeekReasoner = "deepseek-reasoner"
)

// ----------------------------------------------------------------------------
// OpenAI-compatible request/response types (DeepSeek uses the same format)
// ----------------------------------------------------------------------------

// chatMessage represents a single message in the OpenAI chat format.
// It supports all roles: system, user, assistant (with optional tool_calls), and tool.
type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type assistantMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Tools       []chatTool    `json:"tools,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	MaxTokens   int32         `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream"`
}

type chatChoice struct {
	Index   int              `json:"index"`
	Message assistantMessage `json:"message"`
	Delta   assistantMessage `json:"delta"` // used in streaming
}

type usageInfo struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usageInfo    `json:"usage"`
}

// ----------------------------------------------------------------------------
// Model implementation
// ----------------------------------------------------------------------------

// Model implements model.LLM for the DeepSeek API.
type Model struct {
	modelName  string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// Option is a functional option for configuring a DeepSeek Model.
type Option func(*Model)

// WithAPIKey sets the DeepSeek API key explicitly.
// If not set, the DEEPSEEK_API_KEY environment variable is used.
func WithAPIKey(key string) Option {
	return func(m *Model) { m.apiKey = key }
}

// WithBaseURL overrides the DeepSeek API base URL.
// Useful for pointing at a local proxy or compatible endpoint.
func WithBaseURL(url string) Option {
	return func(m *Model) { m.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient replaces the default HTTP client (e.g. for testing).
func WithHTTPClient(c *http.Client) Option {
	return func(m *Model) { m.httpClient = c }
}

// New creates a new DeepSeek model that satisfies model.LLM.
//
//	m := deepseek.New("deepseek-chat", deepseek.WithAPIKey("sk-..."))
func New(modelName string, opts ...Option) (*Model, error) {
	m := &Model{
		modelName:  modelName,
		baseURL:    DefaultBaseURL,
		httpClient: http.DefaultClient,
	}
	for _, o := range opts {
		o(m)
	}
	// Fall back to environment variable if no key was supplied.
	if m.apiKey == "" {
		m.apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if m.apiKey == "" {
		return nil, fmt.Errorf("deepseek: API key is required; set DEEPSEEK_API_KEY or use WithAPIKey()")
	}
	return m, nil
}

// Name returns the model identifier, satisfying model.LLM.
func (m *Model) Name() string { return m.modelName }

// GenerateContent translates an ADK LLMRequest into a DeepSeek chat completion
// call and streams back LLMResponse objects, satisfying model.LLM.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Determine the actual model name to use (callback may override req.Model).
		modelName := m.modelName
		if req.Model != "" {
			modelName = req.Model
		}

		// Build the messages slice from ADK's genai.Content list.
		messages, err := contentsToMessages(req)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: building messages: %w", err))
			return
		}

		// Build optional tool definitions.
		tools := buildTools(req)

		// Extract generation config values.
		var temperature float32
		var maxTokens int32
		if req.Config != nil {
			if req.Config.Temperature != nil {
				temperature = *req.Config.Temperature
			}
			if req.Config.MaxOutputTokens != 0 {
				maxTokens = req.Config.MaxOutputTokens
			}
		}

		// Always use non-streaming mode. When ADK requests streaming (the
		// default for console/web launchers), DeepSeek sends SSE chunks, but
		// our adapter does not yet accumulate incremental tool-call deltas
		// across chunks. That causes function-call responses to be yielded
		// as partial, which prevents ADK from executing the tools and
		// triggers "TODO: last event is not final". Forcing non-streaming
		// gives us a single complete response that handles tool calls
		// correctly.
		_ = stream

		dsReq := chatRequest{
			Model:       modelName,
			Messages:    messages,
			Tools:       tools,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			Stream:      false,
		}

		body, err := json.Marshal(dsReq)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: marshalling request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			m.baseURL+"/chat/completions",
			bytes.NewReader(body),
		)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: creating HTTP request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: HTTP request failed: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			errBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("deepseek: API error %d: %s", resp.StatusCode, string(errBody)))
			return
		}

		m.handleNonStreamingResponse(resp.Body, modelName, yield)
	}
}

// handleNonStreamingResponse parses a standard (non-SSE) chat completion response.
func (m *Model) handleNonStreamingResponse(
	body io.Reader,
	modelName string,
	yield func(*model.LLMResponse, error) bool,
) {
	var dsResp chatResponse
	if err := json.NewDecoder(body).Decode(&dsResp); err != nil {
		yield(nil, fmt.Errorf("deepseek: decoding response: %w", err))
		return
	}
	if len(dsResp.Choices) == 0 {
		yield(nil, fmt.Errorf("deepseek: empty choices in response"))
		return
	}

	choice := dsResp.Choices[0]
	llmResp := buildLLMResponse(choice.Message, modelName, dsResp.Usage, false, true)
	yield(llmResp, nil)
}

// handleStreamingResponse parses a Server-Sent Events (SSE) stream.
func (m *Model) handleStreamingResponse(
	body io.Reader,
	modelName string,
	yield func(*model.LLMResponse, error) bool,
) {
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}

		var chunk chatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			yield(nil, fmt.Errorf("deepseek: decoding stream chunk: %w", err))
			return
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		isLast := choice.Delta.Content == "" && len(choice.Delta.ToolCalls) == 0
		llmResp := buildLLMResponse(choice.Delta, modelName, chunk.Usage, !isLast, isLast)
		if !yield(llmResp, nil) {
			return
		}
	}
	if err := scanner.Err(); err != nil {
		yield(nil, fmt.Errorf("deepseek: reading stream: %w", err))
	}
}

// ----------------------------------------------------------------------------
// Translation helpers: ADK ↔ DeepSeek
// ----------------------------------------------------------------------------

// contentsToMessages converts genai.Content items into OpenAI-style chat messages.
func contentsToMessages(req *model.LLMRequest) ([]chatMessage, error) {
	var msgs []chatMessage

	// Inject system instruction if present.
	if req.Config != nil && req.Config.SystemInstruction != nil {
		text := extractText(req.Config.SystemInstruction)
		if text != "" {
			msgs = append(msgs, chatMessage{Role: "system", Content: text})
		}
	}

	for _, c := range req.Contents {
		if c == nil || len(c.Parts) == 0 {
			continue
		}

		role := c.Role
		if role == "model" {
			role = "assistant"
		}
		if role == "" {
			role = "user"
		}

		// Collect text, function calls, and function responses from all parts.
		var textParts []string
		var fnCalls []*genai.FunctionCall
		var fnResponses []*genai.FunctionResponse

		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			switch {
			case p.Text != "":
				textParts = append(textParts, p.Text)
			case p.FunctionCall != nil:
				fnCalls = append(fnCalls, p.FunctionCall)
			case p.FunctionResponse != nil:
				fnResponses = append(fnResponses, p.FunctionResponse)
			}
		}

		// Emit function responses as separate "tool" messages.
		for _, fr := range fnResponses {
			content := ""
			if fr.Response != nil {
				respBytes, _ := json.Marshal(fr.Response)
				content = string(respBytes)
			}
			msgs = append(msgs, chatMessage{
				Role:       "tool",
				ToolCallID: fr.ID,
				Content:    content,
			})
		}

		// Build the main message for this content entry.
		msg := chatMessage{Role: role}

		if len(textParts) > 0 {
			msg.Content = strings.Join(textParts, "\n")
		}

		if len(fnCalls) > 0 {
			msg.Role = "assistant"
			msg.ToolCalls = make([]toolCall, len(fnCalls))
			for i, fc := range fnCalls {
				argsJSON, _ := json.Marshal(fc.Args)
				msg.ToolCalls[i] = toolCall{
					ID:   fc.ID,
					Type: "function",
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{
						Name:      fc.Name,
						Arguments: string(argsJSON),
					},
				}
			}
		}

		// Skip empty user/model messages that only contained function responses
		// (those were already emitted as separate tool messages).
		if msg.Content == "" && len(fnCalls) == 0 {
			continue
		}
		msgs = append(msgs, msg)
	}

	// OpenAI requires the last message to be user or tool (not assistant
	// with tool_calls). Add a fallback if needed.
	if len(msgs) > 0 {
		last := msgs[len(msgs)-1]
		if last.Role == "assistant" && len(last.ToolCalls) > 0 {
			msgs = append(msgs, chatMessage{
				Role:    "user",
				Content: "Please continue.",
			})
		}
	}
	return msgs, nil
}

// extractText pulls plain text from a genai.Content (handles multi-part).
func extractText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// buildTools converts ADK tool definitions into the OpenAI function-calling format.
// Tool declarations are stored in req.Config.Tools as []*genai.Tool.
func buildTools(req *model.LLMRequest) []chatTool {
	if req.Config == nil || len(req.Config.Tools) == 0 {
		return nil
	}
	var tools []chatTool
	for _, t := range req.Config.Tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			// Marshal the parameters schema to JSON. ADK functiontool
			// stores it in ParametersJsonSchema (with proper custom
			// marshaling including the "type" field). Fall back to
			// Parameters for tools that set it directly.
			paramsJSON := json.RawMessage(`{}`)
			switch {
			case fd.ParametersJsonSchema != nil:
				raw, err := json.Marshal(fd.ParametersJsonSchema)
				if err != nil {
					continue
				}
				paramsJSON = json.RawMessage(raw)
			case fd.Parameters != nil:
				raw, err := json.Marshal(fd.Parameters)
				if err != nil {
					continue
				}
				paramsJSON = json.RawMessage(raw)
			}
			tools = append(tools, chatTool{
				Type: "function",
				Function: toolFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  paramsJSON,
				},
			})
		}
	}
	return tools
}

// buildLLMResponse converts a DeepSeek assistant message into an ADK LLMResponse.
func buildLLMResponse(
	msg assistantMessage,
	modelName string,
	usage usageInfo,
	partial bool,
	turnComplete bool,
) *model.LLMResponse {
	resp := &model.LLMResponse{
		ModelVersion: modelName,
		Partial:      partial,
		TurnComplete: turnComplete,
	}

	// Populate usage metadata if present.
	if usage.TotalTokens > 0 {
		resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     usage.PromptTokens,
			CandidatesTokenCount: usage.CompletionTokens,
			TotalTokenCount:      usage.TotalTokens,
		}
	}

	var parts []*genai.Part

	// Text content.
	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}

	// Tool calls → FunctionCall parts.
	for _, tc := range msg.ToolCalls {
		var args map[string]any
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			},
		})
	}

	if len(parts) > 0 {
		resp.Content = &genai.Content{
			Role:  "model",
			Parts: parts,
		}
	}

	return resp
}
