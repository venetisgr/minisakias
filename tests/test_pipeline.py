from __future__ import annotations

import asyncio

from conftest import FakeLLM

from minisakias.llm import LLMError
from minisakias.pipeline import Pipeline, find_citations

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

    router, writer = llm.calls
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
    assert "Δεν βρέθηκαν σχετικές διατάξεις" in llm.calls[-1]["messages"][-1]["content"]
    assert len(llm.calls[-1]["messages"]) == 3                 # the conversation goes along


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


def test_articles_are_capped(settings, corpus, monkeypatch):
    import dataclasses
    small = dataclasses.replace(settings, max_articles=1, max_article_chars=40)
    pipeline = Pipeline(small, corpus, FakeLLM())
    chosen = pipeline.retrieve({"articles": [{"code": "ΠΚ", "number": "308"}, {"code": "ΠΚ", "number": "114"}],
                                "queries": [], "codes": []}, "x")
    assert [a.id for a in chosen] == ["ΠΚ-308"]
    assert "το άρθρο συνεχίζεται" in pipeline.excerpts(chosen)


def test_citations_in_an_answer(corpus):
    text = "Βλ. [ΠΚ 308], [Π.Κ. 114 παρ. 1], [ΠΚ 386, ΚΠολΔ 624], [ΠΚ 9999], [1], [σημείωση] και ΠΚ 386 χωρίς αγκύλες."
    assert [a.id for a in find_citations(text, corpus)] == ["ΠΚ-308", "ΠΚ-114", "ΠΚ-386", "ΚΠολΔ-624"]
