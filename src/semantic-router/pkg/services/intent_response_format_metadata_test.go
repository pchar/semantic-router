package services

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func TestIntentRequestFactsExposeResponseFormat(t *testing.T) {
	for _, kind := range []string{"json_object", "json_schema"} {
		facts := requestFactsForIntent(nil, json.RawMessage(`{"type":"`+kind+`"}`), classification.RequestContextEstimate{})
		if facts.Metadata[intentResponseFormatMetadataKey] != kind {
			t.Fatalf("%s: metadata %v", kind, facts.Metadata)
		}
	}
	for _, raw := range []string{`{"type":"text"}`, ``, `not json`} {
		facts := requestFactsForIntent(nil, json.RawMessage(raw), classification.RequestContextEstimate{})
		if _, ok := facts.Metadata[intentResponseFormatMetadataKey]; ok {
			t.Fatalf("%q must not set %s: %v", raw, intentResponseFormatMetadataKey, facts.Metadata)
		}
	}
	client := requestFactsForIntent(map[string]string{intentResponseFormatMetadataKey: "client"}, json.RawMessage(`{"type":"json_schema"}`), classification.RequestContextEstimate{})
	if client.Metadata[intentResponseFormatMetadataKey] != "client" {
		t.Fatalf("a client metadata key must not be overwritten: %v", client.Metadata)
	}
}
