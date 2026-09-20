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
