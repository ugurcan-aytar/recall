package embed

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// APIProvider is the wire protocol the API embedder talks. The empty
// string means "no API embedder; use local". RECALL_EMBED_PROVIDER may
// only be the empty string ("local"), "openai", or "voyage" — anything
// else falls back to local with a warning printed by the caller.
type APIProvider string

const (
	ProviderLocal  APIProvider = ""        // local GGUF; not API
	ProviderOpenAI APIProvider = "openai"  // text-embedding-3-small (default model)
	ProviderVoyage APIProvider = "voyage"  // voyage-3-lite (default model)
)

// Default model names per provider. Override via APIEmbedderOptions.Model.
const (
	DefaultOpenAIModel = "text-embedding-3-small"
	DefaultVoyageModel = "voyage-3-lite"
)

// API endpoints. Override via APIEmbedderOptions.BaseURL for tests or for
// proxies / Azure / self-hosted compatibles.
const (
	defaultOpenAIBaseURL = "https://api.openai.com/v1"
	defaultVoyageBaseURL = "https://api.voyageai.com/v1"
)

// APIBatchSize is the number of texts sent per HTTP request. 100 keeps
// payloads under each provider's per-request token limit while still
// amortising network round-trips.
const APIBatchSize = 100

// MaxAPIRetries is the cap for backoff retries on 429 / 5xx.
const MaxAPIRetries = 4

// APIEmbedderOptions configures an APIEmbedder. APIKey is read from the
// matching environment variable when left empty.
type APIEmbedderOptions struct {
	Provider   APIProvider // required
	APIKey     string      // "" ⇒ $OPENAI_API_KEY or $VOYAGE_API_KEY
	Model      string      // "" ⇒ provider default
	BaseURL    string      // "" ⇒ provider default; useful for tests / proxies
	Dimensions int         // OpenAI only: passes `dimensions` to truncate; 0 ⇒ store.EmbeddingDimensions
	BatchSize  int         // 0 ⇒ APIBatchSize
	HTTPClient *http.Client
}

// apiEmbedder implements [Embedder] over an HTTP API.
type apiEmbedder struct {
	opts       APIEmbedderOptions
	client     *http.Client
	dims       int
	closed     bool
	modelLabel string
}

// ResolveAPIProvider reads $RECALL_EMBED_PROVIDER. Anything other than
// "openai" / "voyage" returns ProviderLocal so the local embedder remains
// the default in every reachable code path. The default is local and
// must remain local — see CLAUDE.md design principles.
func ResolveAPIProvider() APIProvider {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RECALL_EMBED_PROVIDER"))) {
	case "openai":
		return ProviderOpenAI
	case "voyage":
		return ProviderVoyage
	default:
		return ProviderLocal
	}
}

// NewAPIEmbedder returns an [Embedder] backed by an HTTP provider. Returns
// an error when the provider is unknown or the API key is missing.
//
// This constructor is opt-in: it is reached only when RECALL_EMBED_PROVIDER
// is set explicitly to "openai" or "voyage". The CLI never proposes API
// embedders to users on its own.
func NewAPIEmbedder(opts APIEmbedderOptions) (Embedder, error) {
	if opts.Provider == ProviderLocal {
		return nil, errors.New("api embedder: provider must be 'openai' or 'voyage'")
	}

	switch opts.Provider {
	case ProviderOpenAI:
		if opts.Model == "" {
			opts.Model = DefaultOpenAIModel
		}
		if opts.BaseURL == "" {
			opts.BaseURL = defaultOpenAIBaseURL
		}
		if opts.APIKey == "" {
			opts.APIKey = os.Getenv("OPENAI_API_KEY")
		}
		if opts.APIKey == "" {
			return nil, errors.New("api embedder: $OPENAI_API_KEY is required when RECALL_EMBED_PROVIDER=openai")
		}
		if opts.Dimensions == 0 {
			// Default to whatever the store expects so vectors fit vec0.
			opts.Dimensions = 768
		}
	case ProviderVoyage:
		if opts.Model == "" {
			opts.Model = DefaultVoyageModel
		}
		if opts.BaseURL == "" {
			opts.BaseURL = defaultVoyageBaseURL
		}
		if opts.APIKey == "" {
			opts.APIKey = os.Getenv("VOYAGE_API_KEY")
		}
		if opts.APIKey == "" {
			return nil, errors.New("api embedder: $VOYAGE_API_KEY is required when RECALL_EMBED_PROVIDER=voyage")
		}
		// voyage-3-lite has fixed 512 dim. Caller may override with a model
		// that supports output_dimension via opts.Dimensions; 0 means "let
		// provider pick". We surface the actual dim from the first response.
	default:
		return nil, fmt.Errorf("api embedder: unknown provider %q", opts.Provider)
	}

	if opts.BatchSize <= 0 {
		opts.BatchSize = APIBatchSize
	}
	client := opts.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}

	return &apiEmbedder{
		opts:       opts,
		client:     client,
		dims:       opts.Dimensions, // may be 0 until first response
		modelLabel: providerModelLabel(opts.Provider, opts.Model, opts.Dimensions),
	}, nil
}

// providerModelLabel returns "openai:text-embedding-3-small@768d" /
// "voyage:voyage-3-lite". Stored verbatim in the metadata table so that a
// switch between providers (or even between dimensions of the same model)
// is visible to reconcileModelName.
func providerModelLabel(provider APIProvider, model string, dims int) string {
	if dims > 0 {
		return string(provider) + ":" + model + "@" + fmt.Sprintf("%dd", dims)
	}
	return string(provider) + ":" + model
}

// ---- Embedder interface ----------------------------------------------------

func (a *apiEmbedder) Embed(texts []string) ([][]float32, error) {
	if a.closed {
		return nil, errors.New("api embedder is closed")
	}
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += a.opts.BatchSize {
		end := i + a.opts.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := a.embedBatch(texts[i:end])
		if err != nil {
			return nil, fmt.Errorf("batch %d..%d: %w", i, end, err)
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (a *apiEmbedder) EmbedSingle(text string) ([]float32, error) {
	v, err := a.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

func (a *apiEmbedder) Dimensions() int {
	return a.dims
}

func (a *apiEmbedder) ModelName() string {
	return a.modelLabel
}

func (a *apiEmbedder) Close() error {
	a.closed = true
	return nil
}

// ---- HTTP plumbing ---------------------------------------------------------

// embedRequest is the wire body. Voyage adds `input_type`; OpenAI adds
// `dimensions`. Both fields use omitempty so the JSON only contains
// what's relevant to each provider.
type embedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"` // OpenAI
	InputType  string   `json:"input_type,omitempty"` // Voyage
}

type embedResponseItem struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type embedResponse struct {
	Data []embedResponseItem `json:"data"`
	// Error fields used by both providers when status != 200.
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

// embedBatch sends one request and parses the response. Implements
// exponential backoff on 429 / 5xx up to MaxAPIRetries.
func (a *apiEmbedder) embedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := embedRequest{
		Model: a.opts.Model,
		Input: texts,
	}
	switch a.opts.Provider {
	case ProviderOpenAI:
		if a.opts.Dimensions > 0 {
			reqBody.Dimensions = a.opts.Dimensions
		}
	case ProviderVoyage:
		// Default to "document" — chunks are documents in our pipeline.
		// Query-side embedding goes through the same path; the qmd-style
		// "task: search result | query: …" prefix already signals intent.
		reqBody.InputType = "document"
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := strings.TrimRight(a.opts.BaseURL, "/") + "/embeddings"
	var lastStatus int
	var lastBody []byte

	for attempt := 0; attempt <= MaxAPIRetries; attempt++ {
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+a.opts.APIKey)
		req.Header.Set("User-Agent", "recall-cli")

		resp, err := a.client.Do(req)
		if err != nil {
			// Treat transport errors as retryable.
			if attempt < MaxAPIRetries {
				time.Sleep(backoffDelay(attempt))
				continue
			}
			return nil, fmt.Errorf("post %s: %w", url, err)
		}

		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		lastStatus = resp.StatusCode
		lastBody = body

		if resp.StatusCode == http.StatusOK {
			var parsed embedResponse
			if err := json.Unmarshal(body, &parsed); err != nil {
				return nil, fmt.Errorf("decode response: %w", err)
			}
			if parsed.Error != nil && parsed.Error.Message != "" {
				return nil, fmt.Errorf("%s api error: %s", a.opts.Provider, parsed.Error.Message)
			}
			if len(parsed.Data) != len(texts) {
				return nil, fmt.Errorf("response item count %d != input count %d",
					len(parsed.Data), len(texts))
			}
			out := make([][]float32, len(texts))
			for _, item := range parsed.Data {
				if item.Index < 0 || item.Index >= len(out) {
					return nil, fmt.Errorf("response item index %d out of range", item.Index)
				}
				out[item.Index] = item.Embedding
			}
			// Lock dims after the first successful response so Dimensions()
			// reports what the provider actually returned.
			if a.dims == 0 && len(out) > 0 {
				a.dims = len(out[0])
				a.modelLabel = providerModelLabel(a.opts.Provider, a.opts.Model, a.dims)
			}
			return out, nil
		}

		if shouldRetry(resp.StatusCode) && attempt < MaxAPIRetries {
			time.Sleep(backoffDelay(attempt))
			continue
		}

		return nil, parseHTTPError(a.opts.Provider, resp.StatusCode, body)
	}

	return nil, fmt.Errorf("%s api: gave up after %d retries (last status %d, body=%s)",
		a.opts.Provider, MaxAPIRetries, lastStatus, truncate(string(lastBody), 200))
}

// shouldRetry returns true for transient HTTP statuses.
func shouldRetry(status int) bool {
	if status == http.StatusTooManyRequests {
		return true
	}
	if status >= 500 && status <= 599 {
		return true
	}
	return false
}

// backoffDelay returns 1s, 2s, 4s, 8s … capped at 30s. Indirected through
// a package-level var so tests can swap in a faster schedule.
var backoffDelay = func(attempt int) time.Duration {
	d := time.Second << uint(attempt)
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}

// parseHTTPError decodes the structured error body when present, otherwise
// surfaces the raw payload (truncated). Never echoes the request body so
// API keys cannot leak into logs.
func parseHTTPError(provider APIProvider, status int, body []byte) error {
	var env embedResponse
	if json.Unmarshal(body, &env) == nil && env.Error != nil && env.Error.Message != "" {
		return fmt.Errorf("%s api error (HTTP %d): %s", provider, status, env.Error.Message)
	}
	return fmt.Errorf("%s api error (HTTP %d): %s", provider, status, truncate(string(body), 200))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// Static interface-conformance check. Same pattern as MockEmbedder.
var _ Embedder = (*apiEmbedder)(nil)
