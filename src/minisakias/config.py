"""Configuration from the environment (see .env.example). Nothing here is secret except the provider's API key,
which is read at call time and never logged."""
from __future__ import annotations

import os
from dataclasses import dataclass

# An OpenAI-compatible endpoint per provider. `eu` says whether inference runs inside the EU: a citizen's legal
# problem is personal data — often about crimes or health — so by default the service refuses anything else.
PROVIDERS: dict[str, dict] = {
    "tensorx": {"label": "TensorX (Ιρλανδία)", "base_url": "https://api.tensorx.ai/v1", "key_env": "TENSORX_API_KEY",
                "eu": True},
    # any other OpenAI-compatible server (vLLM, llama.cpp, another host): MINISAKIAS_LLM_BASE_URL + MINISAKIAS_LLM_API_KEY
    "custom": {"label": "Προσαρμοσμένος πάροχος", "base_url": "", "key_env": "MINISAKIAS_LLM_API_KEY", "eu": False},
}
EFFORTS = ("", "none", "low", "medium", "high", "max")
# The writer decides what the citizen is told, so it runs on the larger model; the steps around it only choose
# articles and fix spelling, and run on the small fast one. Measured on live questions (2026-11-17): with the same
# articles in hand the Flash writer invented procedures more often.
DEFAULT_WRITER_MODEL = "z-ai/glm-5.3"
DEFAULT_ROUTER_MODEL = "z-ai/glm-5.3-flash"


class ConfigError(Exception):
    """The deployment is misconfigured; the message says what to set."""


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "off", "false", "no")


def _number(name: str, default, cast=int):
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        return cast(value)
    except ValueError as e:
        raise ConfigError(f"{name}={value!r} is not a number") from e


def _effort(name: str, default: str) -> str:
    value = (os.environ.get(name) or default).strip().lower()
    if value not in EFFORTS:
        raise ConfigError(f"{name}={value!r}: one of {', '.join(e or '(empty)' for e in EFFORTS)}")
    return value


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    key_env: str
    require_eu: bool
    demo: bool
    model: str                 # the writer's model
    router_model: str          # the small model: router, selector, proofreader
    proofread: bool            # correct garbled words in the finished answer
    effort: str                # the writer's `reasoning_effort` ("" = not sent)
    router_effort: str
    temperature: float | None
    top_p: float | None
    max_output: int
    request_timeout: float     # seconds to wait for a reply to begin / for a stream's next chunk
    search_limit: int          # hits kept per search query
    max_candidates: int        # articles offered to the selector
    max_articles: int          # how many articles the writer reads at most
    max_article_chars: int     # longer articles are cut
    max_context_chars: int     # all the articles together
    max_message_chars: int
    max_history: int           # messages of the conversation sent to the models
    rate_per_window: int       # questions per address …
    rate_window_s: int         # … in this many seconds
    rate_per_day: int

    @property
    def api_key(self) -> str:
        return (os.environ.get(self.key_env) or "").strip()

    @property
    def configured(self) -> bool:
        return self.demo or bool(self.api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        provider = (os.environ.get("MINISAKIAS_PROVIDER") or "tensorx").strip().lower()
        if provider not in PROVIDERS:
            raise ConfigError(f"MINISAKIAS_PROVIDER={provider!r}: one of {', '.join(PROVIDERS)}")
        spec = PROVIDERS[provider]
        require_eu = _flag("MINISAKIAS_REQUIRE_EU", True)
        base_url = spec["base_url"]
        if provider == "custom":
            base_url = (os.environ.get("MINISAKIAS_LLM_BASE_URL") or "").strip().rstrip("/")
            if not base_url:
                raise ConfigError("MINISAKIAS_PROVIDER=custom needs MINISAKIAS_LLM_BASE_URL")
        if require_eu and not spec["eu"]:
            raise ConfigError(
                f"provider {provider!r} is not known to run inside the EU and MINISAKIAS_REQUIRE_EU is on. "
                "Set MINISAKIAS_REQUIRE_EU=0 only if you have checked where that endpoint processes data.")
        model = (os.environ.get("MINISAKIAS_MODEL") or DEFAULT_WRITER_MODEL).strip()
        temperature = _number("MINISAKIAS_TEMPERATURE", 0.3, float)
        top_p = _number("MINISAKIAS_TOP_P", 0.95, float)
        return cls(
            provider=provider, base_url=base_url, key_env=spec["key_env"], require_eu=require_eu,
            demo=_flag("MINISAKIAS_DEMO", False),
            model=model, router_model=(os.environ.get("MINISAKIAS_ROUTER_MODEL") or DEFAULT_ROUTER_MODEL).strip(),
            proofread=_flag("MINISAKIAS_PROOFREAD", True),
            effort=_effort("MINISAKIAS_EFFORT", "high"), router_effort=_effort("MINISAKIAS_ROUTER_EFFORT", "low"),
            temperature=None if temperature < 0 else temperature, top_p=None if top_p < 0 else top_p,
            max_output=_number("MINISAKIAS_MAX_OUTPUT", 16000),
            request_timeout=_number("MINISAKIAS_TIMEOUT_S", 120.0, float),
            search_limit=_number("MINISAKIAS_SEARCH_LIMIT", 15),
            max_candidates=_number("MINISAKIAS_MAX_CANDIDATES", 80),
            max_articles=_number("MINISAKIAS_MAX_ARTICLES", 18),
            max_article_chars=_number("MINISAKIAS_MAX_ARTICLE_CHARS", 6000),
            max_context_chars=_number("MINISAKIAS_MAX_CONTEXT_CHARS", 80000),
            max_message_chars=_number("MINISAKIAS_MAX_MESSAGE_CHARS", 4000),
            max_history=_number("MINISAKIAS_MAX_HISTORY", 12),
            rate_per_window=_number("MINISAKIAS_RATE_PER_WINDOW", 6),
            rate_window_s=_number("MINISAKIAS_RATE_WINDOW_S", 300),
            rate_per_day=_number("MINISAKIAS_RATE_PER_DAY", 40),
        )
