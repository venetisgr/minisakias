from __future__ import annotations

import asyncio
import dataclasses

from conftest import FakeLLM

from minisakias.llm import LLMError
from minisakias.pipeline import Pipeline, apply_fixes, find_citations

QUESTION = [{"role": "user", "content": "Με χτύπησε ο γείτονας. Μπορώ να κάνω μήνυση;"}]


def run(pipeline: Pipeline, messages) -> list[dict]:
    async def collect():
        return [event async for event in pipeline.answer(messages)]
    return asyncio.run(collect())


def test_a_question_end_to_end(settings, corpus):
    llm = FakeLLM(plan={"articles": [{"code": "ΠΚ", "number": "308"}], "queries": ["υποβολή έγκλησης προθεσμία"],
                        "codes": ["ΠΚ"]})
    events = run(Pipeline(settings, corpus, llm), QUESTION)
    assert [e["type"] for e in events[:3]] == ["status", "sources", "status"]
    assert events[-1]["type"] == "done"
    sources = [s["id"] for s in events[1]["items"]]
    assert sources[0] == "ΠΚ-308"                              # the article the router named comes first
    assert "ΠΚ-114" in sources                                 # and the search found the deadline
    assert "".join(e["text"] for e in events if e["type"] == "text") == llm.answer
    assert [c["id"] for c in events[-1]["cited"]] == ["ΠΚ-308", "ΠΚ-114"]
    assert events[-1]["cited"][0]["version"].startswith("Κείμενο με τις τροποποιήσεις")

    assert [c["step"] for c in llm.calls] == ["router", "writer", "proofreader"]   # four candidates: nothing to select
    router, writer = llm.step("router")[0], llm.step("writer")[0]
    assert "ΠΚ — Ποινικός Κώδικας" in router["system"]          # the router knows which codes exist
    prompt = writer["messages"][-1]["content"]
    assert "[ΠΚ 308] Σωματική βλάβη" in prompt and "Έκδοση κειμένου: Κείμενο με τις τροποποιήσεις" in prompt
    assert prompt.rstrip().endswith("</μήνυμα_πολίτη>") and QUESTION[0]["content"] in prompt
    assert (router["effort"], writer["effort"]) == (settings.router_effort, settings.effort)


def test_when_the_router_fails_the_question_itself_is_searched(settings, corpus):
    for error in (ValueError("not json"), LLMError("busy", "timeout")):
        llm = FakeLLM(route_error=error)
        events = run(Pipeline(settings, corpus, llm), [{"role": "user", "content": "σωματική βλάβη από γείτονα"}])
        assert events[-1]["type"] == "done"
        assert "ΠΚ-308" in [s["id"] for s in events[1]["items"]]


def test_a_misconfigured_provider_is_an_error_not_a_guess(settings, corpus):
    events = run(Pipeline(settings, corpus, FakeLLM(route_error=LLMError("config", "no key"))), QUESTION)
    assert events[-1]["type"] == "error" and events[-1]["kind"] == "config"
    assert "no key" not in events[-1]["message"]               # the citizen is not shown internals


def test_a_writer_failure_ends_the_stream_with_an_error(settings, corpus):
    events = run(Pipeline(settings, corpus, FakeLLM(write_error=LLMError("busy", "503"))), QUESTION)
    assert events[-1] == {"type": "error", "kind": "busy", "message": events[-1]["message"]}
    events = run(Pipeline(settings, corpus, FakeLLM(answer="  ")), QUESTION)
    assert events[-1]["type"] == "error"                       # an empty answer is a failure too


def test_a_plan_with_nothing_to_read_gives_the_writer_no_articles(settings, corpus):
    llm = FakeLLM(answer="Παρακαλώ!")
    events = run(Pipeline(settings, corpus, llm), [*QUESTION, {"role": "assistant", "content": "…"},
                                                   {"role": "user", "content": "Ευχαριστώ πολύ"}])
    assert events[1] == {"type": "sources", "items": []} and events[-1] == {"type": "done", "cited": []}
    writer = llm.step("writer")[0]
    assert "Δεν βρέθηκαν σχετικές διατάξεις" in writer["messages"][-1]["content"]
    assert len(writer["messages"]) == 3                        # the conversation goes along


def test_a_router_reply_of_the_wrong_shape_does_not_break_the_question(settings, corpus):
    llm = FakeLLM(plan={"articles": "ΠΚ 308", "queries": [None, "  ", "απάτη"], "codes": {"x": 1}})
    events = run(Pipeline(settings, corpus, llm), QUESTION)
    assert [s["id"] for s in events[1]["items"]] == ["ΠΚ-386"]


def test_the_conversation_is_windowed_and_roles_alternate(settings, corpus):
    messages = []
    for i in range(20):
        messages += [{"role": "user", "content": f"ερώτηση {i}"}, {"role": "assistant", "content": f"απάντηση {i}"}]
    messages += [{"role": "user", "content": "πρώτο"}, {"role": "user", "content": "δεύτερο"}]   # after a failed answer
    llm = FakeLLM()
    run(Pipeline(settings, corpus, llm), messages)
    sent = llm.calls[0]["messages"]
    assert len(sent) <= settings.max_history and sent[0]["role"] == "user"
    assert all(a["role"] != b["role"] for a, b in zip(sent, sent[1:]))
    assert sent[-1]["content"] == "πρώτο\n\nδεύτερο"


PLAN = {"articles": [{"code": "ΠΚ", "number": "308"}, {"code": "ΠΚ", "number": "114"}],
        "queries": ["απάτη", "διαταγή πληρωμής"], "codes": []}


def test_the_selector_keeps_what_matters_from_a_wide_net(settings, corpus):
    small = dataclasses.replace(settings, max_articles=2)
    llm = FakeLLM(plan=PLAN, selection=["ΠΚ 114", "[ΚΠολΔ 624]", "ΑΚ 914", "ΠΚ 114", 7, "ΠΚ 386", "ΠΚ 308"])
    events = run(Pipeline(small, corpus, llm), QUESTION)
    # its order, without what was never offered (ΑΚ 914), a repeat or a non-label, cut at the cap
    assert [s["id"] for s in events[1]["items"]] == ["ΠΚ-114", "ΚΠολΔ-624"]
    selector = llm.step("selector")[0]
    listing = selector["messages"][-1]["content"]
    assert "[ΠΚ 308] Σωματική βλάβη — 1. Όποιος προξενεί" in listing and QUESTION[0]["content"] in listing
    assert "έως 2 άρθρα" in selector["system"]
    assert "[ΠΚ 308]" not in llm.step("writer")[0]["messages"][-1]["content"]


def test_without_a_usable_selection_the_search_ranking_stands(settings, corpus):
    small = dataclasses.replace(settings, max_articles=2)
    for llm in (FakeLLM(plan=PLAN), FakeLLM(plan=PLAN, selection=["ΑΚ 914", "τίποτα"])):
        events = run(Pipeline(small, corpus, llm), QUESTION)
        assert [s["id"] for s in events[1]["items"]] == ["ΠΚ-308", "ΠΚ-386"] and events[-1]["type"] == "done"


def test_named_articles_rank_like_search_hits_not_above_them(settings, corpus):
    pipeline = Pipeline(settings, corpus, FakeLLM())
    plan = {"articles": [{"code": "ΚΠολΔ", "number": "624"}, {"code": "ΠΚ", "number": "386"}],
            "queries": ["απάτη ψευδή γεγονότα", "απάτη περιουσία"], "codes": []}
    # ΠΚ 386 was named second, but two searches agree on it: it comes before the article named first
    assert [a.id for a in pipeline.candidates(plan, "x")][:2] == ["ΠΚ-386", "ΚΠολΔ-624"]


def test_articles_are_capped(settings, corpus):
    small = dataclasses.replace(settings, max_articles=1, max_article_chars=40)
    pipeline = Pipeline(small, corpus, FakeLLM())
    chosen = asyncio.run(pipeline.select(QUESTION, pipeline.candidates(PLAN, "x")))
    assert [a.id for a in chosen] == ["ΠΚ-308"]
    assert "το άρθρο συνεχίζεται" in pipeline.excerpts(chosen)


def test_garbled_words_are_corrected_after_the_answer(settings, corpus):
    answer = "Στείλτε συνταγμένη επιστολή [ΠΚ 308] για τους α unpaid λογαριασμούς μέσα σε 3 μήνες."
    llm = FakeLLM(answer=answer, fixes=[{"wrong": "συνταγμένη επιστολή", "right": "συστημένη επιστολή"},
                                        {"wrong": "α unpaid λογαριασμούς", "right": "απλήρωτους λογαριασμούς"}])
    events = run(Pipeline(settings, corpus, llm), QUESTION)
    assert [e["type"] for e in events[-2:]] == ["replace", "done"]
    assert events[-2]["text"] == "Στείλτε συστημένη επιστολή [ΠΚ 308] για τους απλήρωτους λογαριασμούς μέσα σε 3 μήνες."
    assert "".join(e["text"] for e in events if e["type"] == "text") == answer     # the stream itself is untouched
    assert llm.step("proofreader")[0]["messages"] == [{"role": "user", "content": answer}]


def test_proofreading_never_fails_or_changes_a_clean_answer(settings, corpus):
    for llm in (FakeLLM(), FakeLLM(proof_error=LLMError("busy", "503")), FakeLLM(proof_error=ValueError("not json")),
                FakeLLM(fixes="όχι λίστα"), FakeLLM(fixes=[{"wrong": "δεν υπάρχει στο κείμενο", "right": "x"}])):
        events = run(Pipeline(settings, corpus, llm), QUESTION)
        assert [e["type"] for e in events[-2:]] == ["text", "done"]
    off = FakeLLM(fixes=[{"wrong": "έγκληση", "right": "μήνυση"}])
    events = run(Pipeline(dataclasses.replace(settings, proofread=False), corpus, off), QUESTION)
    assert not off.step("proofreader") and "replace" not in [e["type"] for e in events]


def test_a_correction_cannot_change_what_the_answer_says():
    text = "Η προθεσμία είναι 3 μήνες [ΠΚ 114] και δεν παρατείνεται. Η εγκληση υποβάλλεται εγγράφως."
    refused = [{"wrong": "3 μήνες", "right": "6 μήνες"},                     # a number
               {"wrong": "ΠΚ 114", "right": "ΠΚ 115"},                        # inside a citation
               {"wrong": "[ΠΚ 114]", "right": "[ΠΚ 115]"},
               {"wrong": "δεν παρατείνεται", "right": "παρατείνεται"},       # a negation
               {"wrong": "παρατείνεται", "right": "δεν παρατείνεται"},
               {"wrong": "κληση", "right": "κλήση"},                          # part of a word
               {"wrong": "Η προθεσμία", "right": "Η προθεσμία"},             # no change
               {"wrong": "μήνες", "right": "μήνες, κατά τη γνώμη μου, που είναι πολύ λίγο και άδικο για όλους"},
               "όχι αντικείμενο", {"wrong": 3, "right": None}]
    assert apply_fixes(text, refused) == (text, 0)
    fixed, applied = apply_fixes(text, refused + [{"wrong": "εγκληση", "right": "έγκληση"}])
    assert applied == 1 and fixed == text.replace("εγκληση", "έγκληση")


def test_citations_in_an_answer(corpus):
    text = "Βλ. [ΠΚ 308], [Π.Κ. 114 παρ. 1], [ΠΚ 386, ΚΠολΔ 624], [ΠΚ 9999], [1], [σημείωση] και ΠΚ 386 χωρίς αγκύλες."
    assert [a.id for a in find_citations(text, corpus)] == ["ΠΚ-308", "ΠΚ-114", "ΠΚ-386", "ΚΠολΔ-624"]
