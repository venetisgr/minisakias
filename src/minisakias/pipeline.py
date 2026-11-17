"""One question, start to finish: the router names articles and searches, the corpus supplies the candidates, the
selector keeps the ones that matter, the writer answers from them. Yields the events the website renders as they
happen.

    {"type": "status", "stage": "routing" | "writing"}
    {"type": "sources", "items": [...]}          the articles the writer was given
    {"type": "text", "text": "..."}              the answer, as it is written
    {"type": "replace", "text": "..."}           the whole answer again, with garbled words corrected (if any were)
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
from .prompts import (PROOFREADER_SYSTEM, ROUTER_SYSTEM, SELECTOR_SYSTEM, WRITER_SYSTEM, selector_user_message,
                      writer_user_message)

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


_NEGATION = re.compile(r"(?<![α-ωά-ώ])(δεν|δε|μη|μην|όχι|ούτε|χωρίς)(?![α-ωά-ώ])", re.I)
_LETTER = re.compile(r"[^\W\d_]")


def apply_fixes(text: str, fixes: list) -> tuple[str, int]:
    """The answer with the proofreader's corrections applied — only the ones that cannot change what it says: a
    correction never touches a number, a citation or a negation, and replaces whole words only."""
    applied = 0
    for fix in fixes[:20]:
        if not isinstance(fix, dict):
            continue
        wrong, right = fix.get("wrong"), fix.get("right")
        if not (isinstance(wrong, str) and isinstance(right, str) and wrong.strip() and right.strip()):
            continue
        wrong, right = wrong.strip(), right.strip()
        if (wrong == right or len(wrong) > 80 or len(right) > 100 or len(right) > 3 * len(wrong) + 10
                or any(c in wrong + right for c in "[]\n")
                or re.findall(r"\d", wrong) != re.findall(r"\d", right)
                or len(_NEGATION.findall(wrong)) != len(_NEGATION.findall(right))):
            continue
        out, at, changed = [], 0, False
        while (i := text.find(wrong, at)) >= 0:
            before, after = text[i - 1: i], text[i + len(wrong): i + len(wrong) + 1]
            inside_word = (_LETTER.match(before) and _LETTER.match(wrong[0])) or \
                          (_LETTER.match(after) and _LETTER.match(wrong[-1]))
            inside_citation = text.rfind("[", 0, i) > text.rfind("]", 0, i)
            out.append(text[at:i])
            if inside_word or inside_citation:
                out.append(wrong)
            else:
                out.append(right)
                changed = True
            at = i + len(wrong)
        if changed:
            text = "".join(out) + text[at:]
            applied += 1
    return text, applied


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
            "articles": [a for a in _clean_list(plan.get("articles"), 10) if isinstance(a, dict)],
            "queries": [q.strip() for q in _clean_list(plan.get("queries"), 6) if isinstance(q, str) and q.strip()],
            "codes": [c for c in _clean_list(plan.get("codes"), 8) if isinstance(c, str)],
        }

    # --- step 2: candidates ------------------------------------------------------------------------------------------
    def candidates(self, plan: dict | None, question: str) -> list[Article]:
        """A wide net: the articles the router named and the best of each search, merged by reciprocal rank so that
        an article several lists agree on comes first. Lexical search finds the right article and a dozen that only
        share its words; `select` reads them and keeps the ones that matter."""
        if plan is None:
            plan = {"articles": [], "queries": [question[:400]], "codes": []}
        scores: dict[str, float] = {}
        by_id: dict[str, Article] = {}
        # The articles the router named are one more ranked list, not a list apart: a model remembers the famous
        # numbers (ΠΚ 308, ΑΚ 914) and guesses the rest, and a wrong number brings an unrelated article. Ranked like
        # a search, a right guess that a search also finds rises and a wrong one sinks below the searches' best.
        rank = 0
        for named in plan["articles"]:
            article = self.corpus.find(str(named.get("code", "")), str(named.get("number", "")))
            if article and article.id not in by_id:
                by_id[article.id] = article
                scores[article.id] = 1.0 / (20 + rank)
                rank += 1
        prefer = {abbr for c in plan["codes"] if (abbr := self.corpus.code_abbr(c))}
        for query in plan["queries"]:
            hits = self.corpus.search(query, limit=self.settings.search_limit, prefer=prefer)
            for rank, (article, _) in enumerate(hits):
                by_id[article.id] = article
                scores[article.id] = scores.get(article.id, 0.0) + 1.0 / (20 + rank)
        ranked = sorted(by_id.values(), key=lambda a: -scores[a.id])
        return ranked[: self.settings.max_candidates]

    # --- step 3: the ones that matter --------------------------------------------------------------------------------
    @staticmethod
    def _listing(candidates: list[Article]) -> str:
        """One line per candidate: its label, heading and how it begins — enough to tell what it is about."""
        lines = []
        for a in candidates:
            snippet = " ".join(a.text[:260].split())
            lines.append(f"[{a.label}] {a.heading + ' — ' if a.heading else ''}{snippet}")
        return "\n".join(lines)

    async def select(self, history: list[dict], candidates: list[Article]) -> list[Article]:
        """The candidates the writer should read, most important first, chosen by the model from their headings and
        opening lines. If the choice cannot be had, the candidates' own ranking stands."""
        cap = self.settings.max_articles
        chosen: list[Article] = []
        if len(candidates) > cap:
            question = history[-1]["content"]
            messages = [*history[:-1], {"role": "user", "content": selector_user_message(question, self._listing(candidates))}]
            try:
                reply = await self.llm.complete_json(SELECTOR_SYSTEM.format(max_articles=cap), messages,
                                                     model=self.settings.router_model,
                                                     effort=self.settings.router_effort, max_tokens=4000)
                offered = {a.id: a for a in candidates}
                for label in _clean_list(reply.get("articles"), 2 * cap):
                    m = _CITATION.search(label) if isinstance(label, str) else None
                    article = self.corpus.find(m.group(1), m.group(2)) if m else None
                    if article and article.id in offered and article not in chosen:
                        chosen.append(article)
            except LLMError as e:
                if e.kind == "config":
                    raise
                log.warning("selector call failed (%s); keeping the search ranking", e)
            except ValueError as e:
                log.warning("selector reply was not JSON (%s); keeping the search ranking", e)
        if not chosen:
            chosen = candidates
        kept, used = [], 0
        for article in chosen[:cap]:
            size = min(len(article.text), self.settings.max_article_chars)
            if used + size > self.settings.max_context_chars:
                break
            kept.append(article)
            used += size
        return kept

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

    # --- step 5: the words ---------------------------------------------------------------------------------------------
    async def proofread(self, answer: str) -> str:
        """The answer with its garbled words corrected (open models sometimes write «συνταγμένη επιστολή» for
        «συστημένη», or drop a foreign word into a Greek sentence). Never fails a question: on any trouble the answer
        stands as written."""
        try:
            reply = await self.llm.complete_json(PROOFREADER_SYSTEM, [{"role": "user", "content": answer}],
                                                 model=self.settings.router_model,
                                                 effort=self.settings.router_effort, max_tokens=6000)
        except (LLMError, ValueError) as e:
            log.warning("proofreading skipped (%s)", e)
            return answer
        fixed, applied = apply_fixes(answer, _clean_list(reply.get("fixes"), 40))
        if applied:
            log.info("proofreading corrected %d word(s)", applied)
        return fixed

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
            articles = await self.select(history, self.candidates(plan, question))
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
            if self.settings.proofread:
                fixed = await self.proofread(answer)
                if fixed != answer:
                    answer = fixed
                    yield {"type": "replace", "text": answer}
            yield {"type": "done", "cited": self.describe(find_citations(answer, self.corpus))}
        except LLMError as e:
            log.error("question failed: %s (%s)", e.kind, e)
            yield {"type": "error", "kind": e.kind, "message": ERROR_MESSAGES.get(e.kind, ERROR_MESSAGES["failed"])}
