package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponseFormatIsExposedToMetadataSignals(t *testing.T) {
	user := []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "list the pods"}}}}
	for _, kind := range []llmprotocol.OutputFormatKind{llmprotocol.OutputJSONObject, llmprotocol.OutputJSONSchema} {
		snapshot := extractSemanticRequestSignals(&llmprotocol.Request{Messages: user, OutputFormat: llmprotocol.OutputFormat{Kind: kind}})
		if snapshot.Metadata[responseFormatMetadataKey] != string(kind) {
			t.Fatalf("%s: metadata %v", kind, snapshot.Metadata)
		}
	}
	plain := extractSemanticRequestSignals(&llmprotocol.Request{Messages: user, OutputFormat: llmprotocol.OutputFormat{Kind: llmprotocol.OutputText}})
	if _, ok := plain.Metadata[responseFormatMetadataKey]; ok {
		t.Fatalf("plain text output must not set %s: %v", responseFormatMetadataKey, plain.Metadata)
	}
	client := extractSemanticRequestSignals(&llmprotocol.Request{
		Messages: user, Metadata: map[string]string{responseFormatMetadataKey: "client-value"},
		OutputFormat: llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONSchema},
	})
	if client.Metadata[responseFormatMetadataKey] != "client-value" {
		t.Fatalf("a client metadata key must not be overwritten: %v", client.Metadata)
	}
}
