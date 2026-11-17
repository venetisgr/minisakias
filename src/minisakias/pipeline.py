"""One question, start to finish: the router names articles and searches, the corpus supplies their text, the
writer answers from them. Yields the events the website renders as they happen.

    {"type": "status", "stage": "routing" | "writing"}
    {"type": "sources", "items": [...]}          the articles the writer was given
    {"type": "text", "text": "..."}              the answer, as it is written
    {"type": "done", "cited": [...]}             the articles the answer actually cites
    {"type": "error", "kind": "...", "message": "..."}
"""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from .config import Settings
from .corpus import Article, Corpus
from .llm import LLM, LLMError
from .prompts import ROUTER_SYSTEM, WRITER_SYSTEM, writer_user_message

log = logging.getLogger("minisakias")

# «[ΠΚ 308]», «[ΑΚ 914, ΑΚ 932]», «[ΚΠΔ 51 παρ. 2]» — a bracket that begins like a citation
_BRACKET = re.compile(r"\[([^\[\]\n]{2,80})\]")
_CITATION = re.compile(r"([A-Za-zΑ-Ωα-ωάέήίόύώ.]{1,12})\s+(\d+\s*[Α-Ωα-ω]{0,2})(?![\dΑ-Ωα-ω])")

ERROR_MESSAGES = {
    "config": "Η υπηρεσία δεν έχει ρυθμιστεί σωστά αυτή τη στιγμή. Δοκιμάστε ξανά αργότερα.",
    "busy": "Το μοντέλο δεν απάντησε εγκαίρως — πιθανώς έχει μεγάλο φόρτο. Δοκιμάστε ξανά σε λίγο.",
    "failed": "Κάτι πήγε στραβά στη σύνταξη της απάντησης. Δοκιμάστε ξανά.",
}


def find_citations(text: str, corpus: Corpus) -> list[Article]:
    """The corpus articles an answer cites in brackets, in order of first mention."""
    found: dict[str, Article] = {}
    for bracket in _BRACKET.findall(text):
        for code, number in _CITATION.findall(bracket):
            article = corpus.find(code, number)
            if article:
                found.setdefault(article.id, article)
    return list(found.values())


def _clean_list(value, limit: int) -> list:
    return value[:limit] if isinstance(value, list) else []


class Pipeline:
    def __init__(self, settings: Settings, corpus: Corpus, llm: LLM):
        self.settings, self.corpus, self.llm = settings, corpus, llm
        self._router_system = ROUTER_SYSTEM.format(catalogue=corpus.catalogue())

    # --- step 1: which articles ------------------------------------------------------------------------------------
    async def route(self, messages: list[dict]) -> dict | None:
        """The router's plan — {"articles", "queries", "codes"} — or None when it could not be had (the search then
        falls back to the citizen's own words). A provider that is down or misconfigured is raised, not hidden."""
        try:
            plan = await self.llm.complete_json(self._router_system, messages, model=self.settings.router_model,
                                                effort=self.settings.router_effort, max_tokens=4000)
        except LLMError as e:
            if e.kind == "config":
                raise
            log.warning("router call failed (%s); searching with the question's own words", e)
            return None
        except ValueError as e:
            log.warning("router reply was not JSON (%s); searching with the question's own words", e)
            return None
        return {
            "articles": [a for a in _clean_list(plan.get("articles"), 15) if isinstance(a, dict)],
            "queries": [q.strip() for q in _clean_list(plan.get("queries"), 6) if isinstance(q, str) and q.strip()],
            "codes": [c for c in _clean_list(plan.get("codes"), 8) if isinstance(c, str)],
        }

    # --- step 2: their text ----------------------------------------------------------------------------------------
    def retrieve(self, plan: dict | None, question: str) -> list[Article]:
        """The articles the writer reads: the ones the router named, then the best of each search, merged by
        reciprocal rank so that an article several searches agree on comes first."""
        if plan is None:
            plan = {"articles": [], "queries": [question[:400]], "codes": []}
        scores: dict[str, float] = {}
        by_id: dict[str, Article] = {}
        for named in plan["articles"]:
            article = self.corpus.find(str(named.get("code", "")), str(named.get("number", "")))
            if article:
                by_id[article.id] = article
                scores[article.id] = scores.get(article.id, 0.0) + 1.0     # named outright: ahead of any search hit
        prefer = {abbr for c in plan["codes"] if (abbr := self.corpus.code_abbr(c))}
        for query in plan["queries"]:
            for rank, (article, _) in enumerate(self.corpus.search(query, limit=8, prefer=prefer)):
                by_id[article.id] = article
                scores[article.id] = scores.get(article.id, 0.0) + 1.0 / (20 + rank)
        ranked = sorted(by_id.values(), key=lambda a: -scores[a.id])
        chosen, used = [], 0
        for article in ranked[: self.settings.max_articles]:
            size = min(len(article.text), self.settings.max_article_chars)
            if used + size > self.settings.max_context_chars:
                break
            chosen.append(article)
            used += size
        return chosen

    def excerpts(self, articles: list[Article]) -> str:
        """The chosen articles as the writer reads them: grouped by code, each code with the note that says which
        version of its text this is."""
        blocks, cap = [], self.settings.max_article_chars
        for abbr, code in self.corpus.codes.items():
            mine = [a for a in articles if a.code == abbr]
            if not mine:
                continue
            lines = [f"### {abbr} — {code['name']}", f"Έκδοση κειμένου: {code['note']}"]
            for a in mine:
                text = a.text if len(a.text) <= cap else a.text[:cap] + " […το άρθρο συνεχίζεται· εδώ δίνεται η αρχή του]"
                title = f"[{a.label}]" + (f" {a.heading}" if a.heading else "")
                lines.append(f"\n{title}\n{text}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def describe(self, articles: list[Article]) -> list[dict]:
        return [a.public(self.corpus.codes.get(a.code)) for a in articles]

    # --- the whole question ----------------------------------------------------------------------------------------
    async def answer(self, messages: list[dict]) -> AsyncIterator[dict]:
        """`messages`: the conversation, oldest first, ending with the citizen's new message."""
        merged: list[dict] = []
        for m in messages:                                     # a failed answer leaves two questions in a row;
            if merged and merged[-1]["role"] == m["role"]:     # chat templates want the roles to alternate
                merged[-1]["content"] += "\n\n" + m["content"]
            else:
                merged.append(dict(m))
        history = merged[-self.settings.max_history:]
        while history and history[0]["role"] != "user":       # a window cut must still begin with the citizen
            history = history[1:]
        question = history[-1]["content"]
        try:
            yield {"type": "status", "stage": "routing"}
            plan = await self.route(history)
            articles = self.retrieve(plan, question)
            yield {"type": "sources", "items": self.describe(articles)}
            yield {"type": "status", "stage": "writing"}
            writer_messages = [*history[:-1],
                               {"role": "user", "content": writer_user_message(question, self.excerpts(articles))}]
            parts = []
            async for text in self.llm.stream(WRITER_SYSTEM, writer_messages, model=self.settings.model,
                                              effort=self.settings.effort, max_tokens=self.settings.max_output):
                parts.append(text)
                yield {"type": "text", "text": text}
            answer = "".join(parts)
            if not answer.strip():
                raise LLMError("failed", "the writer returned no text")
            yield {"type": "done", "cited": self.describe(find_citations(answer, self.corpus))}
        except LLMError as e:
            log.error("question failed: %s (%s)", e.kind, e)
            yield {"type": "error", "kind": e.kind, "message": ERROR_MESSAGES.get(e.kind, ERROR_MESSAGES["failed"])}
