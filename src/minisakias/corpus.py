"""The law corpus: the articles of the Greek codes built by tools/sync_laws.py, with lookup by citation and a
lexical search.

No vector search: the router model writes its queries in the law's own wording, so a BM25 index over stemmed Greek
is enough — and it runs in-process, with nothing to host and nothing leaving the machine.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "laws"

_TOKEN = re.compile(r"[a-zα-ω]+")
# words that carry no legal meaning; matched after normalisation (no accents, final sigma folded)
_STOPWORDS = frozenset("""
ο η το οι τα του τησ των τον την τουσ τισ τα ενα ενασ μια μιασ εναν και κι η ειτε ουτε μητε αλλα ομωσ ενω αν εαν
να θα δεν μη μην σε στο στη στον στην στα στουσ στισ στων με για απο προσ κατα μετα παρα αντι υπο επι δια ωσ εωσ
μεχρι χωρισ που οπου οπωσ οταν οτι ποιοσ ποια ποιο οποιοσ οποια οποιο οποιου οποιασ οποιον οποιαν οποιων οσο οσα
αυτοσ αυτη αυτο αυτου αυτησ αυτων αυτον αυτην αυτα αυτοι αυτεσ εκεινοσ εκεινη εκεινο ειναι ηταν εχει εχουν ειχε
μπορει μπορουν πρεπει καθε αλλοσ αλλη αλλο αλλου αλλων καθωσ επισησ μονο τοτε πιο πολυ εφοσον διοτι επειδη γιατι
τι πωσ μου σου μασ σασ εγω εσυ εμεισ εσεισ εκτοσ εντοσ μεσα εξω πριν αφου ηδη ακομη δηλαδη οχι ναι παρ εδ περ αρθρο
αρθρου αρθρα αρθρων
""".split())
# endings stripped to reach a stem, longest first; «-ση / -σεις / -σεως» all fold to «-σ» so that
# μίσθωση · μισθώσεις · μισθώσεως meet
_SIGMA_ENDINGS = ("σεωσ", "σεων", "σεισ", "σησ", "ση")
_ENDINGS = ("ομενων", "ομενοσ", "ομενου", "ομενεσ", "ομενη", "ομενο", "ομενα", "ουμενοσ", "ουμενη", "ουμενο",
            "ματων", "ματοσ", "ματα", "οντασ", "ωντασ", "ουσα", "ηκαν", "ουνται", "ονται", "ειται", "εται",
            "ουμε", "ουν", "ειασ", "ιων", "εισ", "ουσ", "ιασ", "ων", "εσ", "ασ", "ησ", "οσ", "ου", "οι", "ον", "ην",
            "αν", "ει", "ια", "ιο", "η", "α", "ο", "ι", "ε", "ω", "υ", "σ")
# a model may write a code's abbreviation with Latin look-alikes («AK» for «ΑΚ»)
_LATIN_TO_GREEK = str.maketrans("ABEZHIKMNOPTYX", "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ")
_CODE_ALIASES = {"ΣΥΝΤΑΓΜΑ": "Σ", "ΣΥΝΤ": "Σ", "ΚΠΟΙΝΔ": "ΚΠΔ", "ΠΟΙΝΚ": "ΠΚ", "ΚΔΔΙΑΔ": "ΚΔΔιαδ", "ΚΠΟΛΔ": "ΚΠολΔ",
                 "ΕΙΣΝΑΚ": "ΕισΝΑΚ", "ΕΙΣΝΚΠΟΛΔ": "ΕισΝΚΠολΔ", "ΚΩΔΔΙΚ": "ΚωδΔικ", "ΦΠΑ": "ΚΦΠΑ"}


def normalise(text: str) -> str:
    """Lower case, no accents, final sigma folded: «Έγκλησης» → «εγκλησησ»."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("ς", "σ")


def stem(word: str) -> str:
    """A rough Greek stem. It only has to make the inflections of one word meet, not be linguistically right."""
    for ending in _SIGMA_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)] + "σ"
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in _TOKEN.findall(normalise(text)) if len(w) > 1 and w not in _STOPWORDS]


def code_key(abbr: str) -> str:
    """«Π.Κ.», «πκ», «AK» (Latin) → the form the lookup table is keyed by."""
    key = unicodedata.normalize("NFD", re.sub(r"[\s.]", "", abbr).upper())
    key = "".join(c for c in key if unicodedata.category(c) != "Mn").translate(_LATIN_TO_GREEK)
    return _CODE_ALIASES.get(key, key).upper()


def number_key(number: str) -> str:
    """«308», «308 παρ. 2», «5α», «0042» → «308», «308», «5Α», «42»."""
    m = re.match(r"^\s*0*(\d+)\s*([Α-Ωα-ωA-Za-z]{1,2})?\b", str(number))
    if not m:
        return ""
    suffix = (m.group(2) or "").upper().translate(_LATIN_TO_GREEK)
    return m.group(1) + suffix


@dataclass(frozen=True)
class Article:
    id: str          # «ΠΚ-308»
    code: str        # «ΠΚ»
    label: str       # «ΠΚ 308» — what an answer cites, in brackets
    number: str
    heading: str
    chapter: str
    text: str

    def public(self, code: dict | None = None) -> dict:
        """What the website gets."""
        out = {"id": self.id, "label": self.label, "code": self.code, "number": self.number, "heading": self.heading}
        if code:
            out.update(code_name=code["name"], version=code["note"])
        return out


class Corpus:
    def __init__(self, articles: list[Article], info: dict):
        self.articles = articles
        self.info = info
        self.codes: dict[str, dict] = {c["abbr"]: c for c in info.get("codes", [])}
        self._by_id = {a.id: a for a in articles}
        self._by_citation = {(code_key(a.code), number_key(a.number)): a for a in articles}
        self._code_keys = {code_key(abbr): abbr for abbr in self.codes}
        self._build_index()

    @classmethod
    def load(cls, directory: Path | None = None) -> "Corpus":
        directory = directory or DATA_DIR
        with (directory / "articles.jsonl").open(encoding="utf-8") as fh:
            articles = [Article(**{k: row[k] for k in Article.__dataclass_fields__})
                        for row in map(json.loads, fh)]
        info = json.loads((directory / "codes.json").read_text(encoding="utf-8"))
        return cls(articles, info)

    # --- lookup ---------------------------------------------------------------------------------------------------
    def get(self, article_id: str) -> Article | None:
        return self._by_id.get(article_id)

    def find(self, code: str, number: str) -> Article | None:
        """The article a citation names, however it was typed («π.κ.», «308 παρ. 2»). None if the corpus lacks it."""
        return self._by_citation.get((code_key(code), number_key(number)))

    def code_abbr(self, code: str) -> str | None:
        return self._code_keys.get(code_key(code))

    def catalogue(self) -> str:
        """The codes on record, one per line, for the router's prompt."""
        return "\n".join(f"- {c['abbr']} — {c['name']} ({c['articles']} άρθρα)" for c in self.codes.values())

    # --- search ---------------------------------------------------------------------------------------------------
    def _build_index(self, k1: float = 1.2, b: float = 0.75) -> None:
        postings: dict[str, dict[int, int]] = {}
        lengths = []
        for i, a in enumerate(self.articles):
            # the heading says what the article is about: it counts three times
            toks = tokens(a.heading) * 3 + tokens(a.chapter) + tokens(a.text)
            lengths.append(len(toks))
            for t in toks:
                bucket = postings.setdefault(t, {})
                bucket[i] = bucket.get(i, 0) + 1
        n = len(self.articles) or 1
        avg = (sum(lengths) / n) or 1.0
        self._k1 = k1
        self._norm = [k1 * (1 - b + b * length / avg) for length in lengths]
        self._postings = {t: list(docs.items()) for t, docs in postings.items()}
        self._idf = {t: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5)) for t, docs in postings.items()}

    def search(self, query: str, limit: int = 8, prefer: set[str] | None = None,
               boost: float = 1.3) -> list[tuple[Article, float]]:
        """The articles that match the query's words best (BM25), best first. Articles of the `prefer` codes get a
        boost — a preference, not a filter: the right article may sit in a code the router did not think of."""
        scores: dict[int, float] = {}
        for t in set(tokens(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, tf in self._postings[t]:
                scores[i] = scores.get(i, 0.0) + idf * tf * (self._k1 + 1) / (tf + self._norm[i])
        if prefer:
            for i in scores:
                if self.articles[i].code in prefer:
                    scores[i] *= boost
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(self.articles[i], s) for i, s in best]
