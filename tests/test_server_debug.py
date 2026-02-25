import json

from mlx_od_moe import server


class _FakeTokenizer:
    def encode(self, text, return_tensors=None):
        del text, return_tensors
        return [151644, 42, 151645]

    def decode(self, token_ids):
        return "".join(f"<{int(i)}>" for i in token_ids)


class _FakeModel:
    expert_store = None

    def __init__(self, output_ids):
        self._output_ids = output_ids

    def generate(self, input_ids, max_tokens, temperature, top_p, stop_token_ids=None):
        del input_ids, max_tokens, temperature, top_p, stop_token_ids
        for token_id in self._output_ids:
            yield token_id


def test_completion_debug_tokens_and_hex():
    prev_model = server.model
    prev_tokenizer = server.tokenizer
    prev_source = server.tokenizer_source
    prev_stop = server.stop_token_ids
    try:
        server.model = _FakeModel([100, 101, 151645])
        server.tokenizer = _FakeTokenizer()
        server.tokenizer_source = "gguf:test:hf_json"
        server.stop_token_ids = [151645]

        client = server.app.test_client()
        resp = client.post(
            "/v1/completions",
            data=json.dumps(
                {
                    "prompt": "<|im_start|>user\nping\n<|im_end|>\n<|im_start|>assistant\n",
                    "max_tokens": 3,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                    "debug_tokens": True,
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["tokens_generated"] == 3
        assert payload["stop_reason"] == "stop_token"
        assert payload["stop_token_id"] == 151645
        assert payload["generated_token_ids"] == [100, 101, 151645]
        assert payload["generated_token_ids_hex"] == ["0x64", "0x65", "0x2505d"]
        assert payload["prompt_token_ids"] == [151644, 42, 151645]
        assert payload["prompt_token_ids_hex"] == ["0x2505c", "0x2a", "0x2505d"]
        assert payload["tokenizer_source"] == "gguf:test:hf_json"
    finally:
        server.model = prev_model
        server.tokenizer = prev_tokenizer
        server.tokenizer_source = prev_source
        server.stop_token_ids = prev_stop


def test_echo_prompt_prepends_prompt_only_when_enabled():
    prev_model = server.model
    prev_tokenizer = server.tokenizer
    prev_source = server.tokenizer_source
    prev_stop = server.stop_token_ids
    try:
        server.model = _FakeModel([200, 201])
        server.tokenizer = _FakeTokenizer()
        server.tokenizer_source = "gguf:test:hf_json"
        server.stop_token_ids = [151645]
        client = server.app.test_client()
        prompt = "<|im_start|>user\nhello\n<|im_end|>\n<|im_start|>assistant\n"

        resp = client.post(
            "/v1/completions",
            data=json.dumps(
                {
                    "prompt": prompt,
                    "max_tokens": 2,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                    "echo_prompt": True,
                    "debug_tokens": True,
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["generated_text_only"] == "<200><201>"
        assert payload["completion"] == prompt + "<200><201>"
    finally:
        server.model = prev_model
        server.tokenizer = prev_tokenizer
        server.tokenizer_source = prev_source
        server.stop_token_ids = prev_stop
