package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func fallbackTestRequest(system string, turns ...string) *RequestContext {
	request := &llmprotocol.Request{}
	if system != "" {
		request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: system}}}}
	}
	for i, turn := range turns {
		role := llmprotocol.RoleUser
		if i%2 == 1 {
			role = llmprotocol.RoleAssistant
		}
		request.Messages = append(request.Messages, llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: turn}}})
	}
	return &RequestContext{Headers: map[string]string{}, SemanticRequest: request}
}

func TestProtectionIdentityFallsBackToConversationStart(t *testing.T) {
	r := &OpenAIRouter{}
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession).RouterLearning.Protection
	cfg.Identity.Fallback = config.RouterLearningIdentityFallbackConversationStart

	first, ok := r.protectionIdentity(fallbackTestRequest("You are Hermes.", "Write a merge function"), cfg)
	if !ok || first.sessionID == "" {
		t.Fatalf("expected a fallback identity, got %+v ok=%v", first, ok)
	}
	later, ok := r.protectionIdentity(fallbackTestRequest("You are Hermes.", "Write a merge function", "def merge(a, b): ...", "Now make it shorter."), cfg)
	if !ok || later.sessionID != first.sessionID {
		t.Fatalf("later turns of one conversation must keep the session: %q vs %q", later.sessionID, first.sessionID)
	}
	other, _ := r.protectionIdentity(fallbackTestRequest("You are Hermes.", "Send the weekly report"), cfg)
	if other.sessionID == first.sessionID {
		t.Fatalf("a different conversation must get a different session")
	}
	// A decision's system_prompt plugin rewrites the instructions before
	// protection runs; the session must not change with them.
	rewritten, _ := r.protectionIdentity(fallbackTestRequest("Decision prompt for code.", "Write a merge function", "def merge(a, b): ...", "Add tests."), cfg)
	if rewritten.sessionID != first.sessionID {
		t.Fatalf("rewritten system instructions must keep the session: %q vs %q", rewritten.sessionID, first.sessionID)
	}
}

func TestProtectionIdentityHeaderWinsAndFallbackIsOptIn(t *testing.T) {
	r := &OpenAIRouter{}
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession).RouterLearning.Protection

	if _, ok := r.protectionIdentity(fallbackTestRequest("", "hello"), cfg); ok {
		t.Fatalf("without fallback a request with no session header has no identity")
	}
	cfg.Identity.Fallback = config.RouterLearningIdentityFallbackConversationStart
	ctx := fallbackTestRequest("", "hello")
	ctx.Headers["x-session-id"] = "client-session"
	identity, ok := r.protectionIdentity(ctx, cfg)
	if !ok || identity.sessionID != "client-session" {
		t.Fatalf("the session header must win over the fallback, got %+v", identity)
	}
	if _, ok := r.protectionIdentity(fallbackTestRequest("system only"), cfg); ok {
		t.Fatalf("a request with no user message has no fallback identity")
	}
}
