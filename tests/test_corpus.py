from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from minisakias.corpus import DATA_DIR, Corpus, code_key, number_key, stem, tokens

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sync_laws", ROOT / "tools" / "sync_laws.py")
sync_laws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_laws)


def test_inflections_of_a_word_meet():
    assert stem("μισθωση") == stem("μισθωσεισ") == stem("μισθωσεωσ")
    assert tokens("Έγκληση") == tokens("εγκλήσεως") == tokens("ΕΓΚΛΗΣΗΣ")
    assert tokens("το άρθρο και η") == []                     # stop words carry nothing


def test_citations_are_read_however_they_are_typed(corpus):
    assert code_key("Π.Κ.") == code_key("πκ") == "ΠΚ"
    assert code_key("AK") == "ΑΚ"                              # Latin look-alikes
    assert number_key("308 παρ. 2") == "308" and number_key("0042") == "42" and number_key("5α") == "5Α"
    assert corpus.find("π.κ.", "308 παρ. 2").id == "ΠΚ-308"
    assert corpus.find("κπολδ", "624").id == "ΚΠολΔ-624"
    assert corpus.find("ΠΚ", "9999") is None
    assert corpus.code_abbr("κπολδ") == "ΚΠολΔ" and corpus.code_abbr("ΧΧΧ") is None


def test_search_ranks_the_article_about_the_words_first(corpus):
    assert corpus.search("σωματική βλάβη")[0][0].id == "ΠΚ-308"
    assert corpus.search("προθεσμία υποβολής έγκλησης")[0][0].id == "ΠΚ-114"
    assert corpus.search("διαταγή πληρωμής")[0][0].id == "ΚΠολΔ-624"
    assert corpus.search("ξυλοκόπος αστροναύτης") == []


def test_a_preferred_code_is_boosted_not_filtered(corpus):
    hits = [a.id for a, _ in corpus.search("έγκληση", prefer={"ΚΠολΔ"})]
    assert "ΠΚ-114" in hits                                    # still found, though its code was not preferred


# --- tools/sync_laws.py: the article files of lexifina/greece-law ----------------------------------------------------
def test_an_inline_heading_is_split_from_paragraph_one():
    a = sync_laws.parse_article("# Άρθρο 308\n\nΣωματική βλάβη 1. Όποιος προξενεί βλάβη τιμωρείται. 2. Απαιτείται "
                                "έγκληση.", "308", ocr=False)
    assert a["number"] == "308" and a["heading"] == "Σωματική βλάβη"
    assert a["text"] == "1. Όποιος προξενεί βλάβη τιμωρείται.\n\n2. Απαιτείται έγκληση."


def test_a_heading_on_the_title_line_runs_on_into_the_body():
    a = sync_laws.parse_article("# Άρθρο 52. - Δικαίωμα προσφυγής\n\nτου εγκαλούντος. 1. Ο εγκαλών έχει δικαίωμα.",
                                "052", ocr=False)
    assert a["number"] == "52" and a["heading"] == "Δικαίωμα προσφυγής του εγκαλούντος"
    assert a["text"].startswith("1. Ο εγκαλών")


def test_chapter_comment_and_ocr_hyphen():
    raw = ("# Άρθρο 914\n\n<!-- text: OCR of official scan -->\n\n_ΚΕΦΑΛΑΙΟ · ΑΔΙΚΟΠΡΑΞΙΕΣ_\n\n"
           "Έννοια. Όποιος ζημιώσει άλλον παρά-- νομα έχει υποχρέωση να τον αποζημιώσει.")
    a = sync_laws.parse_article(raw, "914", ocr=True, sentence_heading=True)
    assert a["chapter"] == "ΚΕΦΑΛΑΙΟ ΑΔΙΚΟΠΡΑΞΙΕΣ" and a["heading"] == "Έννοια"
    assert a["text"] == "Όποιος ζημιώσει άλλον παράνομα έχει υποχρέωση να τον αποζημιώσει."


@pytest.mark.parametrize("title, stem_, number", [
    ("Άρθρο 5Α", "005-a", "5Α"), ("# Άρθρα 3 - 4", "003-004", "3-4"), ("# no title", "187a", "187Α"),
    ("# Άρθρο Πρώτο", "first", None)])
def test_article_numbers(title, stem_, number):
    assert sync_laws.article_number(title, stem_)[0] == number


def test_paragraph_numbers_inside_a_sentence_are_not_split():
    text = "1. Κατά το άρθρο 308 παρ. 2. Η δίωξη ασκείται. 3. Άλλο."
    assert sync_laws.split_paragraphs(text) == text            # «2.» belongs to «παρ. 2», and 3 does not follow 1


def test_the_version_note_says_which_text_this_is():
    enacted = sync_laws.version_of({"history": "as-enacted", "in_force": "1984-10-24", "fek": "164/24.10.1984",
                                    "text": "OCR of official scan"})
    assert enacted["status"] == "as_enacted" and enacted["ocr"] and "δεν έχουν ενσωματωθεί" in enacted["note"]
    assert not sync_laws.version_of({"history": "as-enacted", "text": "PDF text layer (no OCR)"})["ocr"]
    amended = sync_laws.version_of({"history": "as enacted, then amended", "text_as_of": "2019-11-18"})
    assert amended["status"] == "amended_until" and "18.11.2019" in amended["note"]
    assert sync_laws.version_of({"history": "consolidated", "consolidated_to": "updated up to ν. 1/2026"})["status"] \
        == "consolidated"


# --- the corpus that ships -----------------------------------------------------------------------------------------
def test_the_shipped_corpus_holds_the_articles_citizens_ask_about():
    corpus = Corpus.load(DATA_DIR)
    assert len(corpus.articles) > 6000
    for code, number in [("ΠΚ", "308"), ("ΠΚ", "114"), ("ΠΚ", "386"), ("ΚΠΔ", "51"), ("ΑΚ", "914"), ("ΑΚ", "574"),
                         ("ΚΠολΔ", "624"), ("ΚΑΕΔ", "339"), ("ΚΟΚ", "42"), ("Σ", "20")]:
        assert corpus.find(code, number), f"{code} {number} is missing"
    assert corpus.search("σωματική βλάβη έγκληση")[0][0].id == "ΠΚ-308"
    ids = [json.loads(line)["id"] for line in (DATA_DIR / "articles.jsonl").open(encoding="utf-8")]
    assert len(ids) == len(set(ids))
    assert all(code["note"] for code in corpus.codes.values())
