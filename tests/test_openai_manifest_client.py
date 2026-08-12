import asyncio
import sys
from types import SimpleNamespace

from benchmark.scripts import run_openai_manifest


def test_payload_propagates_seed_and_requests_token_ids():
    payload = run_openai_manifest._build_payload(
        "qwen3",
        {
            "prompt": "hello",
            "max_output_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 42,
        },
    )

    assert payload["seed"] == 42
    assert payload["top_k"] == -1
    assert payload["return_token_ids"] is True


def test_stream_result_retains_token_ids_and_hashes():
    chunks = [
        b'data: {"choices":[{"delta":{"content":"A"},"token_ids":[10]}]}\n',
        b'data: {"choices":[{"delta":{"content":"B"},"token_ids":[11]}]}\n',
        b"data: [DONE]\n",
    ]

    class FakeResponse:
        status = 200
        content = None

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def __aiter__(self):
            self._iterator = iter(chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeSession:
        def post(self, _url, *, json):
            assert json["seed"] == 42
            return FakeResponse()

    result = asyncio.run(
        run_openai_manifest.stream_request(
            FakeSession(),
            base_url="http://127.0.0.1:8000",
            model="qwen3",
            request_record={
                "request_id": "r1",
                "prompt": "hello",
                "max_output_tokens": 2,
                "seed": 42,
            },
            tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [99]),
        )
    )

    assert result["output_tokens"] == 2
    assert result["output_token_ids"] == [10, 11]
    assert result["output_token_ids_sha256"] == run_openai_manifest._sha256_json(
        [10, 11]
    )
    assert result["output_text_sha256"] == (
        "38164fbd17603d73f696b8b4d72664d735bb6a7c88577687fd2ae33fd6964153"
    )


def test_single_concurrency_preserves_manifest_request_order(monkeypatch, tmp_path):
    observed = []
    requests = [
        {"request_id": f"r{index}", "prompt": "hello", "max_output_tokens": 1}
        for index in range(4)
    ]

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def fake_stream(_session, *, request_record, **_kwargs):
        observed.append(request_record["request_id"])
        await asyncio.sleep(0)
        return {
            "request_id": request_record["request_id"],
            "ttft_s": 0.1,
            "total_s": 0.2,
            "output_tokens": 1,
            "output_token_ids": [1],
        }

    fake_aiohttp = SimpleNamespace(
        ClientTimeout=lambda **_kwargs: object(),
        ClientSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(run_openai_manifest, "load_requests", lambda *_args, **_kwargs: requests)
    monkeypatch.setattr(run_openai_manifest, "_load_tokenizer", lambda _path: None)
    monkeypatch.setattr(run_openai_manifest, "stream_request", fake_stream)
    args = SimpleNamespace(
        concurrency=1,
        manifest=str(tmp_path / "manifest.jsonl"),
        bucket="mixed_chat",
        max_requests=4,
        tokenizer="",
        request_timeout_s=10,
        base_url="http://127.0.0.1:1",
        model="qwen3",
    )

    summary = asyncio.run(run_openai_manifest.run(args))

    assert observed == ["r0", "r1", "r2", "r3"]
    assert [item["request_id"] for item in summary["per_request"]] == observed
