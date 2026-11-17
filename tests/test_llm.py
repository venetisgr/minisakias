"""The provider calls, against a fake OpenAI-compatible endpoint: what is sent, and how replies are read."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from minisakias.llm import LLM, LLMError, ThinkFilter, extract_json


def sse(*chunks: dict) -> bytes:
    return ("".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def delta(**fields) -> dict:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def completion(content: str) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


def make_llm(settings, handler) -> tuple[LLM, list[dict]]:
    seen: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"),
                     "body": json.loads(request.content)})
        return handler(seen[-1]["body"], len(seen))

    client = AsyncOpenAI(base_url=settings.base_url, api_key="test-key", max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(record)))
    return LLM(settings, client=client), seen


def collect(stream) -> str:
    async def go():
        return "".join([text async for text in stream])
    return asyncio.run(go())


def test_the_writer_stream(settings):
    body = sse(delta(role="assistant", content=""), delta(reasoning_content="σκέφτομαι…"), delta(content="Καλη"),
               delta(content="μέρα"), {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                                       "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                                                "total_tokens": 7}})
    llm, seen = make_llm(settings, lambda b, n: httpx.Response(200, content=body,
                                                                headers={"content-type": "text/event-stream"}))
    text = collect(llm.stream("σύστημα", [{"role": "user", "content": "γεια"}], model="z-ai/glm-5.3-flash",
                              effort="high", max_tokens=100))
    assert text == "Καλημέρα"                                  # the reasoning is not part of the answer
    sent = seen[0]
    assert sent["url"] == "https://api.tensorx.ai/v1/chat/completions" and sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == "z-ai/glm-5.3-flash" and sent["body"]["reasoning_effort"] == "high"
    assert sent["body"]["stream"] is True and sent["body"]["stream_options"] == {"include_usage": True}
    assert (sent["body"]["temperature"], sent["body"]["top_p"]) == (0.3, 0.95)
    assert sent["body"]["messages"][0] == {"role": "system", "content": "σύστημα"}


def test_the_router_reply_is_read_as_json(settings):
    reply = '<think>ας δούμε</think>```json\n{"queries": ["απάτη"], "articles": []}\n```'
    llm, seen = make_llm(settings, lambda b, n: httpx.Response(200, json=completion(reply)))
    plan = asyncio.run(llm.complete_json("s", [{"role": "user", "content": "q"}], model="m", effort="low"))
    assert plan == {"queries": ["απάτη"], "articles": []}
    assert seen[0]["body"]["response_format"] == {"type": "json_object"} and "stream" not in seen[0]["body"]


def test_a_rejected_optional_parameter_is_dropped_and_the_call_repeated(settings):
    def handler(body, n):
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}})
        return httpx.Response(200, json=completion('{"ok": true}'))
    llm, seen = make_llm(settings, handler)
    assert asyncio.run(llm.complete_json("s", [{"role": "user", "content": "q"}], model="m", effort="low")) == {"ok": True}
    assert len(seen) == 2 and "temperature" not in seen[1]["body"] and seen[1]["body"]["max_tokens"] == 2000


@pytest.mark.parametrize("status, kind", [(401, "config"), (429, "busy"), (503, "busy"), (404, "failed")])
def test_provider_errors_become_what_the_citizen_is_told(settings, status, kind):
    llm, _ = make_llm(settings, lambda b, n: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(LLMError) as e:
        collect(llm.stream("s", [{"role": "user", "content": "q"}], model="m", effort="", max_tokens=10))
    assert e.value.kind == kind


def test_no_key_is_a_configuration_error(settings, monkeypatch):
    monkeypatch.delenv("TENSORX_API_KEY")
    with pytest.raises(LLMError) as e:
        asyncio.run(LLM(settings).complete_json("s", [], model="m", effort=""))
    assert e.value.kind == "config"


def test_think_tags_are_filtered_however_they_are_split():
    for chunks in (["<think>μέσα</think>έξω"], ["<th", "ink>μέ", "σα</thi", "nk>έ", "ξω"], ["έ", "ξω"]):
        f = ThinkFilter()
        assert "".join(f.feed(c) for c in chunks) + f.flush() == "έξω"
    f = ThinkFilter()
    assert f.feed("2 < 3 και <b>") + f.flush() == "2 < 3 και <b>"   # other angle brackets pass


def test_extract_json_refuses_what_is_not_an_object():
    assert extract_json('πριν {"a": {"b": 1}} μετά') == {"a": {"b": 1}}
    for bad in ("καμία", "[1, 2]", '{"a": '):
        with pytest.raises(ValueError):
            extract_json(bad)
