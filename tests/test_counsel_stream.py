import time

import httpx

from tests.conftest import parse_sse

API = "/api/v1"


def chat(client, message, session_id=None):
    body = {"message": message, **({"session_id": session_id} if session_id else {})}
    with client.stream("POST", f"{API}/stream_counsel", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse_sse("".join(r.iter_text()))


def test_blocked_until_warmup(client):
    r = client.post(f"{API}/stream_counsel", json={"message": "hi"})
    assert r.status_code == 409


def test_unknown_session_404(warmed):
    r = warmed.post(f"{API}/stream_counsel", json={"message": "hi", "session_id": "nope"})
    assert r.status_code == 404


def test_blank_message_rejected(warmed):
    assert warmed.post(f"{API}/stream_counsel", json={"message": "   "}).status_code == 422


def test_streams_meta_tokens_done(warmed, ollama):
    events = chat(warmed, "I can't decide whether to take the job or stay.")
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("token") == 3
    assert "".join(d["text"] for k, d in events if k == "token") == "I hear how heavy this feels."
    done = events[-1][1]
    assert done["ttft_ms"] is not None and done["session_id"] == events[0][1]["session_id"]
    assert done["emotion"]["primary_emotion"] == "anxiety"


def test_history_persisted_with_emotion(warmed):
    events = chat(warmed, "I feel torn between freedom and safety.")
    sid = events[0][1]["session_id"]
    detail = warmed.get(f"{API}/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["emotion"]["core_conflict"] == "autonomy vs. security"
    assert detail["title"].startswith("I feel torn")


def test_rag_and_history_reach_the_prompt(warmed, ollama):
    first = chat(warmed, "How does Socratic questioning work with my automatic thoughts?")
    sid = first[0][1]["session_id"]
    system = ollama.stream_calls[-1]["messages"][0]["content"]
    assert "Reference material" in system and "Socratic" in system

    chat(warmed, "And what should I do next?", session_id=sid)
    msgs = ollama.stream_calls[-1]["messages"]
    assert msgs[-1] == {"role": "user", "content": "And what should I do next?"}
    assert any(m["role"] == "assistant" and "I hear" in m["content"] for m in msgs)


def test_session_survives_restart_and_continues(make_client, ollama):
    with make_client() as c1:
        c1.post(f"{API}/setup/warmup", json={"topics": ["cbt"]})
        sid = chat(c1, "First message before the restart.")[0][1]["session_id"]
    with make_client() as c2:  # brand-new process state, same SQLite + Qdrant on disk
        assert len(c2.get(f"{API}/sessions/{sid}").json()["messages"]) == 2
        chat(c2, "Second message after the restart.", session_id=sid)
        contents = [m["content"] for m in ollama.stream_calls[-1]["messages"]]
        assert "First message before the restart." in contents


def test_crisis_language_triggers_safety_event(warmed, ollama):
    events = chat(warmed, "Some days I just want to kill myself.")
    kinds = [k for k, _ in events]
    assert "safety" in kinds and kinds.index("safety") < kinds.index("token")
    assert "SAFETY PRIORITY" in ollama.stream_calls[-1]["messages"][0]["content"]


def test_no_safety_event_for_ordinary_message(warmed):
    assert "safety" not in [k for k, _ in chat(warmed, "My roommate and I keep arguing about chores.")]


def test_profile_summary_is_built_in_background(warmed):
    chat(warmed, "I keep going back and forth about leaving my hometown.")
    for _ in range(40):
        profile = warmed.get(f"{API}/profile").json()["summary"]
        if profile:
            break
        time.sleep(0.05)
    assert profile == "Person is torn between autonomy and security."


def test_profile_can_be_edited_and_cleared(warmed):
    assert warmed.put(f"{API}/profile", json={"summary": "Likes long walks."}).json()["summary"] == "Likes long walks."
    assert warmed.delete(f"{API}/profile").status_code == 204
    assert warmed.get(f"{API}/profile").json()["summary"] == ""


def test_delete_session(warmed):
    sid = chat(warmed, "Hello there")[0][1]["session_id"]
    assert warmed.delete(f"{API}/sessions/{sid}").status_code == 204
    assert warmed.get(f"{API}/sessions/{sid}").status_code == 404


def test_ollama_down_yields_error_event(make_client):
    def down(request):
        raise httpx.ConnectError("refused")

    with make_client(transport=httpx.MockTransport(down)) as c:
        c.post(f"{API}/setup/warmup", json={"topics": ["cbt"]})
        events = chat(c, "Are you there?")
        assert events[-1][0] == "error" and "ollama serve" in events[-1][1]["message"]
        assert c.get(f"{API}/health").json()["status"] == "degraded"


def test_emotion_analysis_can_be_disabled(settings, ollama):
    import httpx
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import FakeEmbedder, fake_fetcher

    settings.EMOTION_ANALYSIS_ENABLED = False
    app = create_app(
        settings, embedder=FakeEmbedder(), page_fetcher=fake_fetcher,
        ollama_transport=httpx.MockTransport(ollama.handler),
    )
    with TestClient(app) as c:
        c.post(f"{API}/setup/warmup", json={"topics": ["cbt"]})
        events = chat(c, "I keep replaying an argument with a friend.")
        assert events[-1][0] == "done"
        assert [x for x in ollama.calls if not x.get("stream") and x.get("format")] == []  # no emotion call
        crisis = chat(c, "I want to kill myself.")
        assert "safety" in [k for k, _ in crisis]  # keyword safety net still works


def test_bare_greeting_skips_the_model_and_the_profile(warmed, ollama):
    warmed.put(f"{API}/profile", json={"summary": "Fought with a friend."})
    before = len(ollama.stream_calls)
    events = chat(warmed, "Hello!")
    assert [k for k, _ in events] == ["meta", "token", "done"]
    assert len(ollama.stream_calls) == before  # no LLM call
    sid = events[0][1]["session_id"]
    assert len(warmed.get(f"{API}/sessions/{sid}").json()["messages"]) == 2


def test_profile_is_marked_as_background_only(warmed, ollama):
    warmed.put(f"{API}/profile", json={"summary": "Fought with a friend."})
    chat(warmed, "I need help deciding on a laptop for college.")
    system = ollama.stream_calls[-1]["messages"][0]["content"]
    assert "Fought with a friend." in system and "Do NOT bring these up" in system
    assert "You are an AI" in system


def test_greeting_shortcut_tolerates_typos(warmed, ollama):
    from app.services.counselor_agent import is_greeting

    for ok in ["helllo", "heyyy!", "hii", "Hello there", "good morning", "HELO"]:
        assert is_greeting(ok), ok
    for no in ["hello, I need advice", "hi I fought with my sister", "history", "help me"]:
        assert not is_greeting(no), no


def test_thinking_is_disabled_on_every_llm_request(warmed, ollama):
    chat(warmed, "I can't decide between two job offers.")
    assert ollama.calls, "expected at least one Ollama call"
    assert all(c.get("think") is False for c in ollama.calls)


def test_done_event_reports_llm_timing_and_persona_is_honest(warmed, ollama):
    done = chat(warmed, "I keep second-guessing every choice I make.")[-1][1]
    assert done["llm"] == {
        "prompt_tokens": 123, "prompt_s": 2.0, "load_s": 0.0, "gen_tokens": 20, "gen_tok_per_s": 20.0,
    }
    system = ollama.stream_calls[-1]["messages"][0]["content"]
    assert "AI language model" in system and "running locally" in system


def _system(ollama):
    return ollama.stream_calls[-1]["messages"][0]["content"]


def test_explicit_request_for_help_switches_to_guidance_mode(warmed, ollama):
    chat(warmed, "give me a guide in how to overcome this")
    assert "PRACTICAL GUIDANCE" in _system(ollama)
    chat(warmed, "I felt a bit odd at work today.")
    assert "PRACTICAL GUIDANCE" not in _system(ollama)


def test_repeated_exchanges_nudge_away_from_only_asking_questions(warmed, ollama):
    sid = chat(warmed, "Work has been strange lately.")[0][1]["session_id"]
    chat(warmed, "My manager barely talks to me.", session_id=sid)
    assert "already asked several questions" not in _system(ollama)
    chat(warmed, "I keep wondering what I did wrong.", session_id=sid)
    assert "already asked several questions" in _system(ollama)


def test_crisis_overrides_guidance_mode(warmed, ollama):
    chat(warmed, "I want to kill myself, give me steps to cope")
    system = _system(ollama)
    assert "SAFETY PRIORITY" in system and "PRACTICAL GUIDANCE" not in system


def test_done_event_reports_prep_time(warmed):
    done = chat(warmed, "I feel stuck between two options.")[-1][1]
    assert isinstance(done["prep_ms"], int) and done["prep_ms"] <= done["ttft_ms"]


def test_wants_guidance_detector():
    from app.services.counselor_agent import wants_guidance

    for yes in ["give me a guide in howt o over ocme", "how to overcome this", "What should I do?",
                "any tips?", "help me cope with it", "can you suggest something"]:
        assert wants_guidance(yes), yes
    for no in ["I feel sad today", "my sister and I argued", "he never listens to me", "thanks"]:
        assert not wants_guidance(no), no


def test_style_note_names_phrases_the_model_just_overused():
    from app.services.counselor_agent import style_note

    hist = [
        {"role": "user", "content": "I feel stuck."},
        {"role": "assistant", "content": "It sounds like you're stuck. I wonder what changed?"},
        {"role": "user", "content": "Everything."},
        {"role": "assistant", "content": "It sounds like a lot is going on."},
    ]
    note = style_note(hist)
    assert '"It sounds like"' in note and '"I wonder"' in note and "I hear that" not in note
    assert style_note([]) is None
    assert style_note([{"role": "assistant", "content": "Take a slow breath."}]) is None
    # only the last two replies count
    old = [{"role": "assistant", "content": "I wonder why."}] + [{"role": "assistant", "content": "Okay."}] * 2
    assert style_note(old) is None


def test_style_note_reaches_the_prompt_and_persona_no_longer_teaches_the_phrase(warmed, ollama):
    from app.services.counselor_agent import PERSONA

    assert 'I wonder if' not in PERSONA
    ollama.tokens = ["It sounds like ", "you're tired. ", "I wonder why?"]
    sid = chat(warmed, "Work has been exhausting lately.")[0][1]["session_id"]
    chat(warmed, "I can't switch off in the evenings.", session_id=sid)
    system = ollama.stream_calls[-1]["messages"][0]["content"]
    assert "STYLE:" in system and '"It sounds like"' in system and '"I wonder"' in system
