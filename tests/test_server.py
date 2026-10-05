import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from pagedbatch.server import create_app


@pytest.fixture
def client(engine_factory):
    app = create_app(engine_factory(max_num_seqs=8), model_name="tiny")
    with TestClient(app) as c:
        yield c


def sse_chunks(response) -> list[dict]:
    chunks = []
    for line in response.iter_lines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def test_health_models_and_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
    text = client.get("/metrics").text
    assert "pagedbatch_kv_blocks_total 256" in text and "# TYPE pagedbatch_step_seconds histogram" in text


def test_completion(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5, "temperature": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11}


def test_streaming_matches_non_streaming(client):
    params = {"prompt": "stream", "max_tokens": 6, "temperature": 0, "ignore_eos": True}
    full = client.post("/v1/completions", json=params).json()["choices"][0]["text"]
    with client.stream("POST", "/v1/completions", json={**params, "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = sse_chunks(r)
    assert len(chunks) == 6  # one chunk per sampled token
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"]["role"] == "assistant"
    with client.stream("POST", "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True}) as r:
        chunks = sse_chunks(r)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert len(chunks) == 5


def test_token_prompt_and_validation(client):
    r = client.post("/v1/completions", json={"prompt": [256, 10, 11, 12], "max_tokens": 3, "temperature": 0})
    assert r.status_code == 200 and r.json()["usage"]["prompt_tokens"] == 4
    r = client.post("/v1/completions", json={"prompt": "x" * 600, "max_tokens": 10})
    assert r.status_code == 400 and "max_model_len" in r.json()["detail"]
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
    assert r.status_code == 422


def test_concurrent_requests_share_steps(client):
    # Eight requests of 20 tokens served one at a time would take ~160 steps; batched,
    # the engine needs about 20 decode steps plus a few prefills.
    before = client.get("/metrics").text
    steps_before = int([ln for ln in before.splitlines() if ln.startswith("pagedbatch_steps_total")][0].split()[1])
    body = {"prompt": "batch me", "max_tokens": 20, "temperature": 0, "ignore_eos": True}
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.post("/v1/completions", json=body), range(8)))
    assert all(r.status_code == 200 for r in responses)
    after = client.get("/metrics").text
    steps_after = int([ln for ln in after.splitlines() if ln.startswith("pagedbatch_steps_total")][0].split()[1])
    assert steps_after - steps_before < 80
