from __future__ import annotations

import os

import pytest

from minisakias.config import Settings
from minisakias.corpus import Article, Corpus

CODES = [
    {"abbr": "ΠΚ", "name": "Ποινικός Κώδικας", "articles": 3, "folder": "penal-code", "act": "ν. 4619/2019",
     "fek": "", "source": "https://example.org/pk.pdf", "in_force": "2019-07-01", "status": "amended_until",
     "as_of": "2019-11-18", "ocr": False, "note": "Κείμενο με τις τροποποιήσεις έως 18.11.2019."},
    {"abbr": "ΚΠολΔ", "name": "Κώδικας Πολιτικής Δικονομίας", "articles": 1, "folder": "civil-procedure-code",
     "act": "π.δ. 503/1985", "fek": "182/24.10.1985", "source": "", "in_force": "1985-10-24",
     "status": "as_enacted", "as_of": "1985-10-24", "ocr": True, "note": "Κείμενο όπως αρχικά δημοσιεύθηκε."},
]


def article(code: str, number: str, heading: str, text: str) -> Article:
    return Article(id=f"{code}-{number}", code=code, label=f"{code} {number}", number=number, heading=heading,
                   chapter="", text=text)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus([
        article("ΠΚ", "308", "Σωματική βλάβη", "1. Όποιος προξενεί σε άλλον σωματική κάκωση ή βλάβη της υγείας του "
                "τιμωρείται με φυλάκιση.\n\n2. Για τη δίωξη απαιτείται έγκληση."),
        article("ΠΚ", "114", "Μη υποβολή έγκλησης", "Το αξιόποινο εξαλείφεται αν ο δικαιούχος δεν υποβάλει την "
                "έγκληση μέσα σε τρεις μήνες από την ημέρα που έμαθε για την τέλεση της πράξης."),
        article("ΠΚ", "386", "Απάτη", "Όποιος με σκοπό να αποκομίσει παράνομο περιουσιακό όφελος βλάπτει ξένη "
                "περιουσία πείθοντας κάποιον με ψευδή γεγονότα τιμωρείται με φυλάκιση."),
        article("ΚΠολΔ", "624", "", "Η έκδοση διαταγής πληρωμής μπορεί να ζητηθεί για χρηματική απαίτηση."),
    ], {"codes": CODES, "built": "2026-01-01", "commit": "abc", "repository": "https://example.org/laws"})


@pytest.fixture
def settings(monkeypatch) -> Settings:
    for name in [n for n in os.environ if n.startswith("MINISAKIAS_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("TENSORX_API_KEY", "test-key")
    return Settings.from_env()


class FakeLLM:
    """Stands in for the provider: a fixed plan from the router and a fixed answer from the writer, with every call
    kept for the test to look at."""

    def __init__(self, plan=None, answer="Διώκεται με έγκληση [ΠΚ 308], μέσα σε τρεις μήνες [ΠΚ 114].",
                 route_error=None, write_error=None):
        self.plan, self.answer, self.route_error, self.write_error = plan, answer, route_error, write_error
        self.calls: list[dict] = []

    async def complete_json(self, system, messages, **kw):
        self.calls.append({"step": "router", "system": system, "messages": messages, **kw})
        if self.route_error:
            raise self.route_error
        return self.plan if self.plan is not None else {"articles": [], "queries": [], "codes": []}

    async def stream(self, system, messages, **kw):
        self.calls.append({"step": "writer", "system": system, "messages": messages, **kw})
        if self.write_error:
            raise self.write_error
        for i in range(0, len(self.answer), 7):
            yield self.answer[i: i + 7]
