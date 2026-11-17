"""The web service: the site itself (static files) and the three endpoints behind it.

    POST /api/ask               the conversation in, the answer out as a stream of JSON lines (see pipeline.py)
    GET  /api/articles/{id}     one article's text, for the citation the citizen clicked
    GET  /health                what is deployed: the corpus, the model, whether a provider key is set

Nothing a citizen writes is stored or logged: a conversation lives in the browser and is sent whole with each
question. The only state the service keeps is a per-address counter for the rate limit, in memory.

    uvicorn minisakias.server:app --env-file .env
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import PROVIDERS, ConfigError, Settings
from .corpus import Corpus
from .demo import DemoLLM
from .llm import LLM
from .pipeline import ERROR_MESSAGES, Pipeline

WEB_DIR = Path(__file__).resolve().parent / "web"
log = logging.getLogger("minisakias")

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                                "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=40000)


class AskRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=60)


class RateLimiter:
    """Questions per address: a short window against bursts and a daily cap against cost. In memory, per process —
    enough for one machine; several machines each keep their own count."""

    def __init__(self, per_window: int, window_s: int, per_day: int, clock=time.monotonic):
        self.per_window, self.window_s, self.per_day, self.clock = per_window, window_s, per_day, clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> int:
        """0 when the question may go ahead (and is counted); otherwise the seconds to wait."""
        now, hits = self.clock(), self._hits[key]
        while hits and now - hits[0] >= 86400:
            hits.popleft()
        recent = [t for t in hits if now - t < self.window_s]
        if len(recent) >= self.per_window:
            return max(1, int(self.window_s - (now - recent[0])) + 1)
        if len(hits) >= self.per_day:
            return max(1, int(86400 - (now - hits[0])) + 1)
        hits.append(now)
        if len(self._hits) > 20000:                             # addresses that went quiet
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 86400]:
                del self._hits[k]
        return 0


def client_address(request: Request) -> str:
    """The citizen's address behind Fly's proxy (Fly-Client-IP), behind another proxy, or directly."""
    for header in ("fly-client-ip", "x-forwarded-for"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def wait_text(seconds: int) -> str:
    if seconds < 90:
        return "ένα λεπτό"
    if seconds < 5400:
        return f"{round(seconds / 60)} λεπτά"
    return f"{round(seconds / 3600)} ώρες"


def create_app(settings: Settings | None = None, corpus: Corpus | None = None, llm=None) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.monotonic()
        state["corpus"] = corpus or Corpus.load()
        state["pipeline"] = Pipeline(settings, state["corpus"],
                                     llm or (DemoLLM() if settings.demo else LLM(settings)))
        log.info("corpus ready: %d articles in %d codes (%.1f s)", len(state["corpus"].articles),
                 len(state["corpus"].codes), time.monotonic() - started)
        if not settings.configured:
            log.warning("%s is not set: questions will be refused until it is (or set MINISAKIAS_DEMO=1)",
                        settings.key_env)
        yield

    app = FastAPI(title="Μηνυσάκιας", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    limiter = RateLimiter(settings.rate_per_window, settings.rate_window_s, settings.rate_per_day)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/health")
    async def health():
        c: Corpus = state["corpus"]
        return {
            "status": "ok" if settings.configured else "no_provider_key",
            "version": __version__,
            "demo": settings.demo,
            "provider": {"name": settings.provider, "label": PROVIDERS[settings.provider]["label"],
                         "eu": PROVIDERS[settings.provider]["eu"], "require_eu": settings.require_eu},
            "models": {"writer": settings.model, "writer_effort": settings.effort,
                       "router": settings.router_model, "router_effort": settings.router_effort},
            "corpus": {"articles": len(c.articles), "built": c.info.get("built"), "commit": c.info.get("commit"),
                       "repository": c.info.get("repository"),
                       "codes": [{"abbr": x["abbr"], "name": x["name"], "articles": x["articles"],
                                  "status": x["status"], "note": x["note"]} for x in c.codes.values()]},
        }

    @app.get("/api/articles/{article_id}")
    async def article(article_id: str):
        c: Corpus = state["corpus"]
        found = c.get(article_id)
        if not found:
            raise HTTPException(404, "Το άρθρο δεν υπάρχει στη συλλογή.")
        code = c.codes[found.code]
        return {**found.public(code), "chapter": found.chapter, "text": found.text,
                "act": code["act"], "source": code["source"]}

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request):
        messages = [{"role": m.role, "content": m.content.strip()} for m in body.messages]
        last = messages[-1]
        if last["role"] != "user" or not last["content"]:
            raise HTTPException(422, "Η συνομιλία πρέπει να τελειώνει με μήνυμα του πολίτη.")
        if len(last["content"]) > settings.max_message_chars:
            raise HTTPException(413, f"Το μήνυμα είναι πολύ μεγάλο (έως {settings.max_message_chars} χαρακτήρες).")
        if not settings.configured:
            return JSONResponse({"detail": ERROR_MESSAGES["config"]}, status_code=503)
        wait = limiter.check(client_address(request))
        if wait:
            return JSONResponse(
                {"detail": f"Έχετε κάνει πολλές ερωτήσεις σε λίγο χρόνο. Δοκιμάστε ξανά σε {wait_text(wait)}."},
                status_code=429, headers={"Retry-After": str(wait)})

        async def lines():
            try:
                async for event in state["pipeline"].answer(messages):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception:                                   # never a half-open stream without a last word
                log.exception("unexpected failure while answering")
                yield json.dumps({"type": "error", "kind": "failed", "message": ERROR_MESSAGES["failed"]},
                                 ensure_ascii=False) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/oroi", include_in_schema=False)
    async def terms():
        return FileResponse(WEB_DIR / "oroi.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return create_app()
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}") from e


def __getattr__(name: str):
    # `uvicorn minisakias.server:app` builds the app on first use, so importing this module (tests, tools) reads
    # no environment and loads no corpus
    if name == "app":
        return _app()
    raise AttributeError(name)
