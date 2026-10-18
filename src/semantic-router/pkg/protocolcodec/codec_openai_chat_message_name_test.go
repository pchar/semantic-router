package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatDropsMessageName(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()
	request, envelope, _, err := codec.DecodeRequest([]byte(`{
		"model":"auto",
		"messages":[
			{"role":"system","content":"be brief","name":"sys"},
			{"role":"user","content":"weather in Paris?","name":"bob"},
			{"role":"assistant","content":null,"name":"helper","tool_calls":[
				{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}
			]},
			{"role":"tool","tool_call_id":"c1","name":"get_weather","content":"20C sunny"}
		]
	}`), policy)
	if err != nil {
		t.Fatalf("decode request with messages[].name: %v", err)
	}
	request.Model = "qwen3.6-35b"
	request.Generation++
	body, _, err := codec.EncodeRequest(request, envelope, policy)
	if err != nil {
		t.Fatalf("encode request: %v", err)
	}
	var wire struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	for i, message := range wire.Messages {
		if _, ok := message["name"]; ok {
			t.Fatalf("messages[%d].name forwarded: %s", i, body)
		}
	}
	if !bytes.Contains(body, []byte(`"tool_call_id":"c1"`)) {
		t.Fatalf("tool result lost its tool_call_id: %s", body)
	}
}
