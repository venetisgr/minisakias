"""Calls to the model provider: an OpenAI-compatible chat-completions endpoint (TensorX by default).

Two calls are all the pipeline needs: `complete_json` for the router and `stream` for the writer. The model's
reasoning is never shown: providers send it apart as `reasoning_content` (ignored here), and a self-hosted server
that inlines it as <think>…</think> is filtered.
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import httpx
import openai
from openai import AsyncOpenAI

from .config import Settings

log = logging.getLogger("minisakias")

# parameters not every model or server accepts; a call that is rejected for one of them is repeated without them
OPTIONAL_PARAMS = ("reasoning_effort", "response_format", "stream_options", "temperature", "top_p")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(Exception):
    """A model call failed. `kind` is what the website tells the citizen about: config | busy | failed."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(detail or kind)
        self.kind = kind


class ThinkFilter:
    """Drops <think>…</think> from a stream, whatever way the tags are split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buffer, self._inside = "", False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out = []
        while self._buffer:
            tag = self.CLOSE if self._inside else self.OPEN
            at = self._buffer.find(tag)
            if at >= 0:
                if not self._inside:
                    out.append(self._buffer[:at])
                self._buffer, self._inside = self._buffer[at + len(tag):], not self._inside
                continue
            # keep back a tail that could be the start of a tag
            keep = next((n for n in range(min(len(tag) - 1, len(self._buffer)), 0, -1)
                         if tag.startswith(self._buffer[-n:])), 0)
            if not self._inside:
                out.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self._buffer = ("" if self._inside else self._buffer), ""
        return rest


def extract_json(text: str) -> dict:
    """The JSON object in a model's reply, whether it came bare, fenced in ``` or after some thinking."""
    text = _THINK_BLOCK.sub("", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(text[start: end + 1])
    if not isinstance(obj, dict):
        raise ValueError("the reply's JSON is not an object")
    return obj


def _fail(e: Exception) -> LLMError:
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMError("config", "the provider rejected the API key")
    if isinstance(e, openai.RateLimitError):
        return LLMError("busy", "the provider's rate limit was reached")
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return LLMError("busy", f"the provider could not be reached: {type(e).__name__}")
    if isinstance(e, openai.APIStatusError):
        return LLMError("busy" if e.status_code >= 500 else "failed", f"provider error {e.status_code}")
    return LLMError("failed", type(e).__name__)


def _log_usage(model: str, usage) -> None:
    """Token counts per call — what a question costs — and nothing of what was asked or answered."""
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        log.info("llm call: model=%s input=%s cached=%s output=%s", model, getattr(usage, "prompt_tokens", None),
                 getattr(details, "cached_tokens", None), getattr(usage, "completion_tokens", None))


class LLM:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.settings.api_key:
                raise LLMError("config", f"{self.settings.key_env} is not set")
            self._client = AsyncOpenAI(
                base_url=self.settings.base_url, api_key=self.settings.api_key, max_retries=2,
                timeout=httpx.Timeout(self.settings.request_timeout, connect=10.0))
        return self._client

    def _params(self, model: str, system: str, messages: list[dict], effort: str, max_tokens: int) -> dict:
        s = self.settings
        params: dict = {"model": model, "max_tokens": max_tokens,
                        "messages": [{"role": "system", "content": system}, *messages]}
        if effort:
            params["reasoning_effort"] = effort
        if s.temperature is not None:
            params["temperature"] = s.temperature
        if s.top_p is not None:
            params["top_p"] = s.top_p
        return params

    async def _create(self, **params):
        try:
            return await self.client.chat.completions.create(**params)
        except openai.BadRequestError:
            # a model without reasoning_effort, a server without stream_options …: once more with the bare request
            bare = {k: v for k, v in params.items() if k not in OPTIONAL_PARAMS}
            if len(bare) == len(params):
                raise
            return await self.client.chat.completions.create(**bare)

    async def complete_json(self, system: str, messages: list[dict], *, model: str, effort: str,
                            max_tokens: int = 2000) -> dict:
        """One non-streamed call whose reply is a JSON object. Raises LLMError for a failed call and ValueError
        for a reply that holds no JSON object."""
        params = self._params(model, system, messages, effort, max_tokens)
        params["response_format"] = {"type": "json_object"}
        try:
            response = await self._create(**params)
        except LLMError:
            raise
        except openai.OpenAIError as e:
            raise _fail(e) from e
        _log_usage(model, getattr(response, "usage", None))
        return extract_json(response.choices[0].message.content or "")

    async def stream(self, system: str, messages: list[dict], *, model: str, effort: str,
                     max_tokens: int) -> AsyncIterator[str]:
        """The reply's text as it is written."""
        params = self._params(model, system, messages, effort, max_tokens)
        params.update(stream=True, stream_options={"include_usage": True})
        think = ThinkFilter()
        try:
            response = await self._create(**params)
            async for chunk in response:
                if not chunk.choices:
                    _log_usage(model, getattr(chunk, "usage", None))   # the closing usage chunk
                    continue
                text = think.feed(chunk.choices[0].delta.content or "")
                if text:
                    yield text
        except LLMError:
            raise
        except openai.OpenAIError as e:
            raise _fail(e) from e
        rest = think.flush()
        if rest:
            yield rest
